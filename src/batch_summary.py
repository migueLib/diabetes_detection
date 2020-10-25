def batch_summary(n, epoch, epochs, i, labels, dataset, loss, logger):
    """Logs a summary for each batch """
    if i % n == 0:
        logger.info(f"Epoch [{epoch + 1:3.0f}/{epochs}] - "
                    f"Batch [{i * len(labels):6.0f}/"
                    f"{len(dataset):6.0f}] "
                    f"({(i * len(labels)) / len(dataset) * 100:2.0f}%) - "
                    f"Loss {loss.item():.4f}")
