"""Baseline Instinct memory schema marker.

Incremental DDL is applied by app.db.migrate.run_instinct_migrations() on startup.
This revision exists so Alembic tracks the Instinct layer baseline.
"""

from alembic import op

revision = "001_baseline_instinct"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SELECT 1")


def downgrade() -> None:
    op.execute("SELECT 1")
