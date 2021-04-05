# Steps

1. 两边 cd 到这里

2. `sudo screen docker-compose up`

3. `CREATE DATABASE adnmb_qst_watcher`; 切过去; `./cat_initial_database.sh | pbcopy`，执行
    * drop 已有内容：`DROP SCHEMA public CASCADE; CREATE SCHEMA public;`

4. `./migrate.py ../db.sqlite3 "dbname=$DB user=postgres password=$(cat password.secret) host=$HOST"`
    * ~~`postgresql://postgres$(cat password.secret)@$HOST/$DB`~~