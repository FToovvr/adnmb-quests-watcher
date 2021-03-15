--- TIMESTAMP 时区皆为 UTC

--- 记录以往的运行活动
CREATE TABLE IF NOT EXISTS activity (
    id                      INTEGER,

    -- 本次活动开始执行的时间
    run_at                  TIMESTAMP   NOT NULL,

    -- 本次抓取囊括了从何时开始的串/回应
    fetched_since           TIMESTAMP,
    -- 下次抓取应该截止到的时间，能保证这之前的内容已经抓取到。
    -- 一般应为第一轮抓取时见到的第一串最后回应时间
    ensured_fetched_until   TIMESTAMP,

    is_successful           BOOLEAN     NOT NULL,
    message                 TEXT,

    -- 本次活动的统计信息：
    -- 上传了多少字节
    uploaded_bytes                  INTEGER,
    -- 下载了多少字节
    downloaded_bytes                INTEGER,
    -- 新记录了多少主题串
    newly_recorded_thread_count     INTEGER,
    -- 在已有的主题串中，有多少记录了新的回应
    affected_thread_count           INTEGER,
    -- 新记录了多少回应
    newly_recorded_post_count       INTEGER,
    -- 调用了多少次获取版块页面的 API
    requested_board_page_count      INTEGER,
    -- 调用了多少次获取串页面的 API
    requested_thread_page_count     INTEGER,
    -- 有多少次以登录状态调用了获取串页面的 API
    logged_in_thread_request_count  INTEGER,

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx__activity__run_at
    ON activity(run_at);
CREATE INDEX IF NOT EXISTS idx__activity__is_successful__run_before
    ON activity(is_successful, ensured_fetched_until);

CREATE TABLE IF NOT EXISTS thread (
    id                      INTEGER,
    created_at              TIMESTAMP   NOT NULL,
    user_id                 TEXT        NOT NULL,

    content                 TEXT        NOT NULL,

    -- +
    current_reply_count     INTEGER,

    attachment_base         TEXT,
    attachment_extension    TEXT,
    name                    TEXT,
    email                   TEXT,
    title                   TEXT,

    misc_fields             TEXT,

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx__thread_user_id
    ON thread(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx__thread__created_at__id
    ON thread(created_at, id);

CREATE TABLE IF NOT EXISTS post (
    id                      INTEGER,
    -- +
    parent_thread_id        INTEGER     NOT NULL,
    created_at              TIMESTAMP   NOT NULL,
    user_id                 TEXT        NOT NULL,

    content                 TEXT        NOT NULL,

    attachment_base         TEXT,
    attachment_extension    TEXT,
    name                    TEXT,
    email                   TEXT,
    title                   TEXT,

    misc_fields             TEXT,

    PRIMARY KEY (id),
    FOREIGN KEY (parent_thread_id) REFERENCES thread(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx__post__parent_thread_id__id
    ON post(parent_thread_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS idx__post__parent_thread_id__created_at__id
    ON post(parent_thread_id, created_at, id);

CREATE TABLE IF NOT EXISTS publishing_trace (
    id      INTEGER,
    -- 所发报告的日期，而非发送时的日期
    `date`  DATE    UNIQUE,
    type    TEXT    NOT NULL    CHECK (type IN ('trend', 'new_threads')),
    uuid    TEXT    UNIQUE,

    attempts    NOT NULL    DEFAULT 0,

    to_thread_id            INTEGER,

    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx__publishing_trace__date_type
    ON publishing_trace(`date`, type);

CREATE TABLE IF NOT EXISTS published_post (
    id          INTEGER,
    trace_id    INTEGER NOT NULL,
    -- 指的是报告的第几页，不是在第几页
    page_number INTEGER NOT NULL,

    -- 如果找到所发的串则不为空
    reply_post_id           INTEGER,
    -- 所发的串是第几个回应
    reply_offset            INTEGER,

    PRIMARY KEY (id),
    FOREIGN KEY (trace_id) REFERENCES publishing_trace(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx__published_post__trace_id_type_page_number
    ON published_post(trace_id, page_number);