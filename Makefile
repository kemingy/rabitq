packages := cli disk service

build:
	cargo b

format:
	@cargo +nightly fmt
	@$(foreach package, $(packages), cargo +nightly fmt --package $(package);)

lint:
	@cargo +nightly fmt --check
	@$(foreach package, $(packages), cargo +nightly fmt --package $(package) --check;)
	@cargo clippy -- -D warnings
	@$(foreach package, $(packages), cargo clippy --package $(package) -- -D warnings;)

test:
	@cargo test
