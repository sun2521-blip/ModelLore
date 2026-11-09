"""Errors whose messages are safe to show to API clients."""


class TrainingInputError(ValueError):
    """The dataset or configuration can't be used for training.

    The message describes the problem in terms of the user's own input, such as
    column names or configuration fields. It never includes file paths or cell
    values, so the API can return it as-is. Any other exception raised during
    training gets a generic message instead.
    """
