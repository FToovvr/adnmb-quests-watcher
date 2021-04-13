
clean-logs:
	rm -rf logs/*

generate-requirements:
	pipreqs . --force