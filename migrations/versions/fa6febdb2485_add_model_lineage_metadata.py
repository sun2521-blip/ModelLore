"""add model lineage metadata

Revision ID: fa6febdb2485
Revises: c96c20459e81
Create Date: 2026-10-26 11:08:27.015964

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fa6febdb2485'
down_revision = 'c96c20459e81'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "models",
        sa.Column("lineage_run_id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "models",
        sa.Column("feature_columns", sa.JSON(), nullable=True),
    )
    op.create_index(
        "ix_models_lineage_run_id",
        "models",
        ["lineage_run_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_models_lineage_run_id", table_name="models")
    op.drop_column("models", "feature_columns")
    op.drop_column("models", "lineage_run_id")
