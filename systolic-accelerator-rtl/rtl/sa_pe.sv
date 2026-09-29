// Weight-stationary processing element.
// Holds one int8 weight. Activations move right, partial sums move down.
//   a_out <= a_in
//   p_out <= p_in + a_in * w
// The weight register shifts down the column when w_shift is high, which
// is how the controller preloads a new weight tile.
module sa_pe #(
    parameter int AW = 8,   // activation and weight width
    parameter int PW = 32   // partial sum width
) (
    input  logic                 clk,
    input  logic                 w_shift,
    input  logic signed [AW-1:0] w_in,
    output logic signed [AW-1:0] w_out,
    input  logic signed [AW-1:0] a_in,
    output logic signed [AW-1:0] a_out,
    input  logic signed [PW-1:0] p_in,
    output logic signed [PW-1:0] p_out
);
    logic signed [AW-1:0]   w_q;
    logic signed [2*AW-1:0] prod;

    assign prod  = a_in * w_q;   // both operands signed, 16 bit context
    assign w_out = w_q;

    always_ff @(posedge clk) begin
        if (w_shift) w_q <= w_in;
        a_out <= a_in;
        p_out <= p_in + PW'(prod);
    end
endmodule
