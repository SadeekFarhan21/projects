# All targets assume the local opam switch in this directory:
#   eval $(opam env --switch=. --set-switch)
.PHONY: build test fuzz bench plots verilog verilator verilator-check all clean

build:
	dune build

test:
	dune test --force

fuzz:
	dune exec -- gpu fuzz --n 1000 --seed 42 --out results/fuzz.json

bench:
	dune exec -- gpu bench --out results/bench.csv

plots:
	uv run scripts/plot_bench.py results/bench.csv

verilog:
	dune exec -- gpu emit-verilog generated/gpu.v

verilator: verilog
	cmake -S verilator -B verilator/build -G Ninja -DCMAKE_BUILD_TYPE=Release
	cmake --build verilator/build

verilator-check: verilator
	rm -rf vectors && dune exec -- gpu gen-vectors vectors --n 300 --seed 7
	./verilator/build/gpu_tb vectors/*.in
	dune exec -- gpu compare-vectors vectors --out results/verilator_agreement.json

all: build test fuzz bench plots verilator-check

clean:
	dune clean
	rm -rf generated vectors verilator/build
