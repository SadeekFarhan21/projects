// Fixed delay line of D register stages (D = 0 is a wire).
module sa_delay #(
    parameter int W = 8,
    parameter int D = 1
) (
    input  logic         clk,
    input  logic [W-1:0] d,
    output logic [W-1:0] q
);
    if (D == 0) begin : g_wire
        assign q = d;
    end else begin : g_regs
        logic [W-1:0] stage [D];
        always_ff @(posedge clk) begin
            stage[0] <= d;
            for (int i = 1; i < D; i++) stage[i] <= stage[i-1];
        end
        assign q = stage[D-1];
    end
endmodule
