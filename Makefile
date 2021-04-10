
clean-logs:
	rm -rf logs/*

generate-requirements:
	yes n | pigar --without-referenced-comments -i venv