// N x N grid of weight-stationary PEs.
//   a_left[r]  enters row r at column 0 and flows right
//   psum enters every column at row 0 as zero and flows down
//   p_bot[c]   leaves the bottom of column c
//   w_top[c]   enters column c at row 0 and shifts down when w_shift is high
module sa_array #(
    parameter int N  = 8,
    parameter int AW = 8,
    parameter int PW = 32
) (
    input  logic                 clk,
    input  logic                 w_shift,
    input  logic signed [AW-1:0] w_top  [N],
    input  logic signed [AW-1:0] a_left [N],
    output logic signed [PW-1:0] p_bot  [N]
);
    logic signed [AW-1:0] a_h [N][N+1];  // horizontal activation wires
    logic signed [PW-1:0] p_v [N+1][N];  // vertical partial sum wires
    logic signed [AW-1:0] w_v [N+1][N];  // vertical weight shift wires

    for (genvar r = 0; r < N; r++) begin : g_rowin
        assign a_h[r][0] = a_left[r];
    end
    for (genvar c = 0; c < N; c++) begin : g_colio
        assign p_v[0][c] = '0;
        assign w_v[0][c] = w_top[c];
        assign p_bot[c]  = p_v[N][c];
    end

    for (genvar r = 0; r < N; r++) begin : g_row
        for (genvar c = 0; c < N; c++) begin : g_col
            sa_pe #(.AW(AW), .PW(PW)) u_pe (
                .clk    (clk),
                .w_shift(w_shift),
                .w_in   (w_v[r][c]),
                .w_out  (w_v[r+1][c]),
                .a_in   (a_h[r][c]),
                .a_out  (a_h[r][c+1]),
                .p_in   (p_v[r][c]),
                .p_out  (p_v[r+1][c])
            );
        end
    end
endmodule
