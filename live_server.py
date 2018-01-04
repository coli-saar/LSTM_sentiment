import torch

import utils
import settings
import torchwordemb

from live_sentiment import text2vec

import socket


if __name__ == "__main__":
    # Load model
    model = utils.generate_model_from_settings()
    utils.load_model_params(model, settings.args.load_path)

    # Load glove
    print("Reading word vectors...")
    vocab, vec = torchwordemb.load_glove_text(settings.DATA_KWARGS["glove_path"])
    print("Done!")

    # Start listening for connections
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 8098))
    s.listen(1)
    conn, addr = s.accept()
    print("Connection established")

    while True:
        text = conn.recv(1024).decode("utf-8")
        features = text2vec(text, vocab, vec)
        features = utils.pack_sequence([features])
        (features, lengths) = torch.nn.utils.rnn.pad_packed_sequence(features)
        out = model(features, lengths)

        stars = "{}".format(float(out[0, 0, 0])).encode("utf-8")
        conn.send(stars)
