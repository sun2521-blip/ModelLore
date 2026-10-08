def train(model, X_train, y_train):
    if model is None:
        raise ValueError("Model cannot be None.")
    model.fit(X_train, y_train)
    return model