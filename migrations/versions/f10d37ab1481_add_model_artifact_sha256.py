"""add model artifact sha256

Revision ID: f10d37ab1481
Revises: fa6febdb2485
Create Date: 2026-11-09 22:57:29.866183

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f10d37ab1481'
down_revision: Union[str, Sequence[str], None] = 'fa6febdb2485'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Record the SHA-256 of each model artifact so it can be checked before loading.

    Nullable: models registered before this revision have no recorded hash and
    must be retrained before /predict will load them.
    """
    op.add_column(
        "models",
        sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    """Drop the artifact hash column."""
    # Batch mode rebuilds the table on SQLite, which cannot always drop columns in place.
    with op.batch_alter_table("models") as batch_op:
        batch_op.drop_column("artifact_sha256")
