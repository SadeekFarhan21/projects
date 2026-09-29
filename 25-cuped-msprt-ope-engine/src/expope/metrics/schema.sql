-- Event-log schema for the experimentation engine.
-- One row per fact; nothing is pre-aggregated. Metrics are computed on demand in SQL.

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id    VARCHAR   PRIMARY KEY,
    control_variant  VARCHAR   NOT NULL,
    treatment_variant VARCHAR  NOT NULL,
    treatment_share  DOUBLE    NOT NULL CHECK (treatment_share > 0 AND treatment_share < 1),
    start_at         TIMESTAMP NOT NULL,
    end_at           TIMESTAMP NOT NULL,
    pre_period_days  INTEGER   NOT NULL DEFAULT 14,
    CHECK (end_at > start_at)
);

-- Randomisation unit is the user. Exactly one assignment per (experiment, user).
CREATE TABLE IF NOT EXISTS assignments (
    experiment_id VARCHAR   NOT NULL,
    user_id       BIGINT    NOT NULL,
    variant       VARCHAR   NOT NULL,
    assigned_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (experiment_id, user_id)
);

-- A user can be exposed (triggered) many times; analysis uses the first exposure.
CREATE TABLE IF NOT EXISTS exposures (
    experiment_id VARCHAR   NOT NULL,
    user_id       BIGINT    NOT NULL,
    exposed_at    TIMESTAMP NOT NULL
);

-- Generic behavioural events. value carries revenue for purchases and is NULL otherwise.
CREATE TABLE IF NOT EXISTS events (
    user_id    BIGINT    NOT NULL,
    event_type VARCHAR   NOT NULL,
    ts         TIMESTAMP NOT NULL,
    value      DOUBLE
);
