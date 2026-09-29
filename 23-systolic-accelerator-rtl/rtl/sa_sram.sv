// Behavioral simple dual port SRAM: one write port, one registered read port.
// Written in the style Yosys infers as ECP5 block RAM (DP16KD) when large.
module sa_sram #(
    parameter int W     = 64,
    parameter int DEPTH = 64,
    parameter int AWID  = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
    input  logic            clk,
    input  logic            we,
    input  logic [AWID-1:0] waddr,
    input  logic [W-1:0]    wdata,
    input  logic            re,
    input  logic [AWID-1:0] raddr,
    output logic [W-1:0]    rdata
);
    logic [W-1:0] mem [DEPTH];
    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        if (re) rdata <= mem[raddr];
    end
endmodule
