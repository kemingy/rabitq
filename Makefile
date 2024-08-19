build:
	cargo b

format:
	@cargo +nightly fmt

lint:
	@cargo +nightly fmt -- --check
	@cargo clippy -- -D warnings

test:
	@cargo test
