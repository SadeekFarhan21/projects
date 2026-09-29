// Weight-stationary int8 systolic array accelerator, one tile command at a time.
//
// A command computes   OUT[0:m, 0:N]  (+)=  A[0:m, 0:N] @ W[0:N, 0:N]
// where A arrives on the activation stream (one row of N int8 per beat),
// W arrives on the weight stream (row k of the tile per beat, k = 0..N-1)
// and OUT is kept in the output buffer as int32.
//
// Command fields
//   cmd_m      rows of A in this command, 1..ADEPTH
//   cmd_load_w 1 = accept N new weight rows and preload them into the array,
//              0 = keep the weights already in the PEs
//   cmd_acc    1 = add the result to the output buffer, 0 = overwrite
//   cmd_drain  1 = stream the m output rows out after compute
//
// Controller phases (strictly sequential in v0)
//   IDLE -> [LOADW] -> LOADA -> [PRELOAD] -> COMPUTE -> [DRAIN] -> IDLE
//
// All ready outputs depend only on registered state, never on a valid input.
module sa_top #(
    parameter int N      = 8,
    parameter int ADEPTH = 64,
    parameter int AW     = 8,
    parameter int PW     = 32,
    localparam int MW    = $clog2(ADEPTH + 1),
    localparam int BAW   = (ADEPTH > 1) ? $clog2(ADEPTH) : 1,
    localparam int WAW   = (N > 1) ? $clog2(N) : 1,
    localparam int CW    = $clog2(ADEPTH + 2 * N + 2) + 1
) (
    input  logic            clk,
    input  logic            rst_n,

    input  logic            cmd_valid,
    output logic            cmd_ready,
    input  logic [MW-1:0]   cmd_m,
    input  logic            cmd_load_w,
    input  logic            cmd_acc,
    input  logic            cmd_drain,

    input  logic            w_valid,
    output logic            w_ready,
    input  logic [N*AW-1:0] w_data,

    input  logic            a_valid,
    output logic            a_ready,
    input  logic [N*AW-1:0] a_data,

    output logic            o_valid,
    input  logic            o_ready,
    output logic [N*PW-1:0] o_data,

    output logic            busy
);
    typedef enum logic [2:0] {
        S_IDLE    = 3'd0,
        S_LOADW   = 3'd1,
        S_LOADA   = 3'd2,
        S_PRELOAD = 3'd3,
        S_COMPUTE = 3'd4,
        S_DRAIN   = 3'd5
    } state_t;

    state_t        state;
    logic [MW-1:0] m_q;
    logic          load_w_q, acc_q, drain_q;
    logic [CW-1:0] cnt;      // phase counter
    logic [MW-1:0] rp;       // drain read pointer
    logic [MW-1:0] oc;       // drain rows popped

    // ------------------------------------------------------------------
    // Handshakes
    // ------------------------------------------------------------------
    assign cmd_ready = (state == S_IDLE);
    assign w_ready   = (state == S_LOADW);
    assign a_ready   = (state == S_LOADA);
    assign busy      = (state != S_IDLE);

    wire cmd_fire = cmd_valid & cmd_ready;
    wire w_fire   = w_valid & w_ready;
    wire a_fire   = a_valid & a_ready;

    // ------------------------------------------------------------------
    // Buffers
    // ------------------------------------------------------------------
    // Weight staging buffer: N rows of N int8
    logic            wb_re;
    logic [WAW-1:0]  wb_raddr;
    logic [N*AW-1:0] wb_rdata;
    sa_sram #(.W(N*AW), .DEPTH(N)) u_wbuf (
        .clk(clk),
        .we(w_fire), .waddr(cnt[WAW-1:0]), .wdata(w_data),
        .re(wb_re), .raddr(wb_raddr), .rdata(wb_rdata)
    );

    // Activation buffer: ADEPTH rows of N int8
    logic            ab_re;
    logic [BAW-1:0]  ab_raddr;
    logic [N*AW-1:0] ab_rdata;
    sa_sram #(.W(N*AW), .DEPTH(ADEPTH)) u_abuf (
        .clk(clk),
        .we(a_fire), .waddr(cnt[BAW-1:0]), .wdata(a_data),
        .re(ab_re), .raddr(ab_raddr), .rdata(ab_rdata)
    );

    // Output buffer: ADEPTH rows of N int32
    logic            ob_we, ob_re;
    logic [BAW-1:0]  ob_waddr, ob_raddr;
    logic [N*PW-1:0] ob_wdata, ob_rdata;
    sa_sram #(.W(N*PW), .DEPTH(ADEPTH)) u_obuf (
        .clk(clk),
        .we(ob_we), .waddr(ob_waddr), .wdata(ob_wdata),
        .re(ob_re), .raddr(ob_raddr), .rdata(ob_rdata)
    );

    // ------------------------------------------------------------------
    // Weight preload: read wbuf rows N-1 .. 0, shift into the array top.
    // PRELOAD lasts N+1 cycles (cnt = 0..N), shifting on cnt = 1..N.
    // ------------------------------------------------------------------
    assign wb_re    = (state == S_PRELOAD) && (cnt < CW'(N));
    assign wb_raddr = WAW'(CW'(N - 1) - cnt);
    wire   w_shift  = (state == S_PRELOAD) && (cnt != '0);

    logic signed [AW-1:0] w_top [N];
    for (genvar c = 0; c < N; c++) begin : g_wtop
        assign w_top[c] = wb_rdata[c*AW +: AW];
    end

    // ------------------------------------------------------------------
    // Compute: issue abuf row cnt for cnt < m. Row t reaches the bottom of
    // the deskew network at cnt = t + 2N and is written to obuf then.
    // ------------------------------------------------------------------
    localparam int LAT = 2 * N;
    wire in_compute = (state == S_COMPUTE);
    assign ab_re    = in_compute && (cnt < CW'(m_q));
    assign ab_raddr = cnt[BAW-1:0];

    logic ab_vld_q;  // ab_rdata holds a row issued last cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) ab_vld_q <= 1'b0;
        else        ab_vld_q <= ab_re;
    end

    // Input skew: row r delayed by r cycles. Invalid slots are zero.
    logic signed [AW-1:0] a_left [N];
    for (genvar r = 0; r < N; r++) begin : g_skew
        logic [AW-1:0] a_src;
        assign a_src = ab_vld_q ? ab_rdata[r*AW +: AW] : '0;
        sa_delay #(.W(AW), .D(r)) u_d (.clk(clk), .d(a_src), .q(a_left[r]));
    end

    logic signed [PW-1:0] p_bot [N];
    sa_array #(.N(N), .AW(AW), .PW(PW)) u_array (
        .clk(clk), .w_shift(w_shift), .w_top(w_top), .a_left(a_left), .p_bot(p_bot)
    );

    // Output deskew: column c delayed by N-1-c cycles.
    logic [N*PW-1:0] res_row;
    for (genvar c = 0; c < N; c++) begin : g_deskew
        sa_delay #(.W(PW), .D(N - 1 - c)) u_d (
            .clk(clk), .d(p_bot[c]), .q(res_row[c*PW +: PW])
        );
    end

    // Accumulate read happens one cycle before the write of the same row.
    wire [CW-1:0] rd_row = cnt - CW'(LAT - 1);
    wire [CW-1:0] wr_row = cnt - CW'(LAT);
    wire acc_rd = in_compute && acc_q && (cnt >= CW'(LAT - 1)) && (rd_row < CW'(m_q));
    wire res_wr = in_compute && (cnt >= CW'(LAT)) && (wr_row < CW'(m_q));

    assign ob_we    = res_wr;
    assign ob_waddr = wr_row[BAW-1:0];
    for (genvar c = 0; c < N; c++) begin : g_accum
        assign ob_wdata[c*PW +: PW] = res_row[c*PW +: PW]
                                    + (acc_q ? ob_rdata[c*PW +: PW] : '0);
    end

    // ------------------------------------------------------------------
    // Drain: obuf -> 2 entry FIFO -> o stream (valid/ready)
    // ------------------------------------------------------------------
    logic            dr_issue, dr_rvld;
    logic [N*PW-1:0] fifo_q [2];
    logic            fifo_rd, fifo_wr;  // entry pointers
    logic [1:0]      fifo_cnt;
    wire             pop  = o_valid & o_ready;
    wire             push = dr_rvld;

    assign dr_issue = (state == S_DRAIN) && (rp < m_q)
                    && ({1'b0, fifo_cnt} + {2'b0, dr_rvld} - {2'b0, pop} < 3'd2);

    assign ob_re    = acc_rd | dr_issue;
    assign ob_raddr = (state == S_DRAIN) ? rp[BAW-1:0] : rd_row[BAW-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) dr_rvld <= 1'b0;
        else        dr_rvld <= dr_issue;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_rd  <= 1'b0;
            fifo_wr  <= 1'b0;
            fifo_cnt <= '0;
        end else begin
            if (push) fifo_wr <= ~fifo_wr;
            if (pop)  fifo_rd <= ~fifo_rd;
            fifo_cnt <= fifo_cnt + {1'b0, push} - {1'b0, pop};
        end
    end
    always_ff @(posedge clk) begin
        if (push) fifo_q[fifo_wr] <= ob_rdata;
    end

    assign o_valid = (fifo_cnt != '0);
`ifdef SA_FAULT_OUT_GLITCH
    // Test only: corrupt o_data while the sink stalls, to prove the
    // handshake assertion fires.
    logic glitch_q;
    always_ff @(posedge clk) glitch_q <= o_valid & ~o_ready;
    assign o_data = fifo_q[fifo_rd] ^ {{(N*PW-1){1'b0}}, glitch_q};
`else
    assign o_data  = fifo_q[fifo_rd];
`endif

    // ------------------------------------------------------------------
    // Controller FSM
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            m_q      <= '0;
            load_w_q <= 1'b0;
            acc_q    <= 1'b0;
            drain_q  <= 1'b0;
            cnt      <= '0;
            rp       <= '0;
            oc       <= '0;
        end else begin
            case (state)
                S_IDLE: if (cmd_fire) begin
                    m_q      <= cmd_m;
                    load_w_q <= cmd_load_w;
                    acc_q    <= cmd_acc;
                    drain_q  <= cmd_drain;
                    cnt      <= '0;
                    state    <= cmd_load_w ? S_LOADW : S_LOADA;
                end
                S_LOADW: if (w_fire) begin
                    if (cnt == CW'(N - 1)) begin
                        cnt   <= '0;
                        state <= S_LOADA;
                    end else cnt <= cnt + 1'b1;
                end
                S_LOADA: if (a_fire) begin
                    if (cnt == CW'(m_q) - 1'b1) begin
                        cnt   <= '0;
                        state <= load_w_q ? S_PRELOAD : S_COMPUTE;
                    end else cnt <= cnt + 1'b1;
                end
                S_PRELOAD: begin
                    if (cnt == CW'(N)) begin
                        cnt   <= '0;
                        state <= S_COMPUTE;
                    end else cnt <= cnt + 1'b1;
                end
                S_COMPUTE: begin
                    if (cnt == CW'(m_q) + CW'(LAT) - 1'b1) begin
                        cnt   <= '0;
                        rp    <= '0;
                        oc    <= '0;
                        state <= drain_q ? S_DRAIN : S_IDLE;
                    end else cnt <= cnt + 1'b1;
                end
                S_DRAIN: begin
                    if (dr_issue) rp <= rp + 1'b1;
                    if (pop) begin
                        if (oc == m_q - 1'b1) state <= S_IDLE;
                        oc <= oc + 1'b1;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // Assertions. Each failure prints and bumps sva_errors, which the
    // cocotb testbench checks at the end of every test.
    // ------------------------------------------------------------------
`ifndef SYNTHESIS
    logic [31:0] sva_errors /* verilator public_flat_rd */;
    logic        past_ok;
    logic            prev_stall;
    logic [N*PW-1:0] prev_o_data;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sva_errors  <= '0;
            past_ok     <= 1'b0;
            prev_stall  <= 1'b0;
            prev_o_data <= '0;
        end else begin
            automatic logic [31:0] nf = '0;
            past_ok     <= 1'b1;
            prev_stall  <= o_valid & ~o_ready;
            prev_o_data <= o_data;
            // A1: o_valid stays high and o_data stays stable while stalled
            if (past_ok && prev_stall && !(o_valid && o_data == prev_o_data)) begin
                $display("SVA A1 FAIL t=%0t o stream dropped valid or changed data under stall", $time);
                nf = nf + 1;
            end
            // A2: legal command sizes
            if (cmd_fire && (cmd_m == '0 || cmd_m > MW'(ADEPTH))) begin
                $display("SVA A2 FAIL t=%0t illegal cmd_m=%0d", $time, cmd_m);
                nf = nf + 1;
            end
            // A3: FIFO never overflows or underflows
            if ((push && !pop && fifo_cnt == 2'd2) || (pop && fifo_cnt == 2'd0)) begin
                $display("SVA A3 FAIL t=%0t output FIFO overflow or underflow", $time);
                nf = nf + 1;
            end
            // A4: o_valid only while draining
            if (o_valid && state != S_DRAIN) begin
                $display("SVA A4 FAIL t=%0t o_valid outside DRAIN", $time);
                nf = nf + 1;
            end
            // A5: at most one ready is high, and only in its phase
            if (32'(cmd_ready) + 32'(w_ready) + 32'(a_ready) > 1) begin
                $display("SVA A5 FAIL t=%0t more than one input ready", $time);
                nf = nf + 1;
            end
            // A6: state encoding is legal
            if (state > S_DRAIN) begin
                $display("SVA A6 FAIL t=%0t illegal state", $time);
                nf = nf + 1;
            end
            // A7: drain never reads past the rows of this command
            if (dr_issue && rp >= m_q) begin
                $display("SVA A7 FAIL t=%0t drain read past m", $time);
                nf = nf + 1;
            end
            sva_errors <= sva_errors + nf;
        end
    end
`endif
endmodule
