
create-new-db:
	rm -f db.sqlite3 && sqlite3 db.sqlite3 < create_db.sql

clean-logs:
	rm -f logs/*