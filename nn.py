import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import (L1,L2) 
from tensorflow.keras.layers import (SimpleRNN, Dense, Conv1D, Conv2D, MaxPooling2D,
                                      Flatten, Bidirectional, LSTM, GRU, Embedding, 
                                      Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D)

# GPU Check
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# use the tokenizer from DistilRoBERTa
tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")


def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length")


def to_bow(example):
    """Converts the sequence of 1-hot vectors into a single many-hot vector"""
    vector = numpy.zeros(shape=(tokenizer.vocab_size,))
    vector[example["input_ids"]] = 1
    return {"input_bow": vector}


def train(model_path="cnn_w3h_etm5_2", train_path="../graduate-project-data/train.csv", dev_path="../graduate-project-data/dev.csv"):

    # load the CSVs into Huggingface datasets to allow use of the tokenizer
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # the labels are the names of all columns except the first
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        # the float here is because F1Score requires floats
        return {"labels": [float(example[l]) for l in labels]}

    # convert text and labels to format expected by model
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    #hf_dataset = hf_dataset.map(to_bow) # For Feed Forward NN

    # convert Huggingface datasets to Tensorflow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="input_ids", # "input_bow" for FF
        label_cols="labels", 
        batch_size=16,
        shuffle=True)
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="input_ids", # input_bow for FF
        label_cols="labels",
        batch_size=16)
    
    # Class weight calculation
    all_ld = np.array([])
    for batch in train_dataset:
        input_data, label_data = batch
        ld = tf.convert_to_tensor(label_data, dtype=tf.float32)
        all_ld = np.vstack([all_ld, ld.numpy()]) if all_ld.size else ld.numpy()

    # You can sum along axis 0 to get the count of each class
    class_counts = np.sum(all_ld, axis=0)

    # total number of samples in your training dataset
    total = len(all_ld)

    # Calculate class weights
    #class_weights = {i: total / (2.0 * count) for i, count in enumerate(class_counts)}
    class_weights = {
    0: total / (2.0 * class_counts[0]) * 4,  # Admiration
    1: total / (2.0 * class_counts[1]) * 2, # Amusement
    2: total / (2.0 * class_counts[2]), # Gratitude
    3: total / (2.0 * class_counts[3]) * 3, # Love
    4: total / (2.0 * class_counts[4]), # Pride
    5: total / (2.0 * class_counts[5]), # Relief
    6: total / (2.0 * class_counts[6]),  # Remorse
    }
    
    # define a model with a single fully connected layer
    # Default model
    #model = tf.keras.Sequential([tf.keras.layers.Dense(units=len(labels), input_dim=tokenizer.vocab_size, activation='sigmoid')])
    
    # RNN model
    # model = Sequential()
    # model.add(Embedding(input_dim=tokenizer.vocab_size, output_dim=128, input_length=64))
    # model.add(SimpleRNN(128, activation='relu', input_shape=(tokenizer.vocab_size,), return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(len(labels), activation='sigmoid'))

    # Bidiretional GRU model
    # model = Sequential()
    # model.add(Embedding(input_dim=tokenizer.vocab_size, output_dim=128, input_length=64))
    # model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=(tokenizer.vocab_size,)))
    # model.add(Dropout(0.5)) 
    # model.add(Dense(64, activation='relu')) #model.add(Dense(64, activation='relu', kernel_regularizer=L2(0.01)))
    # model.add(Dropout(0.5))
    # model.add(Flatten()) #model.add(GlobalAveragePooling1D())
    # model.add(Dense(len(labels), activation='sigmoid'))

    # Hyperband tuned Bidirectional GRU model 
    # learning_rate: 0.001
    # threshold: 0.4
    # batch_size: 64
    # epochs: 10
    # model = Sequential()
    # model.add(Embedding(input_dim=tokenizer.vocab_size, output_dim=128, input_length=64))
    # model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=(tokenizer.vocab_size,)))
    # model.add(Dropout(0.1)) 
    # model.add(Dense(64, activation='relu')) #model.add(Dense(64, activation='relu', kernel_regularizer=L2(0.01)))
    # model.add(Dropout(0.1))
    # model.add(Flatten()) #model.add(GlobalAveragePooling1D()) 
    # model.add(Dense(len(labels), activation='sigmoid'))

    # CNN model
    model = Sequential()
    model.add(Embedding(input_dim=tokenizer.vocab_size, output_dim=256, input_length=64))
    model.add(Conv1D(128, kernel_size=2, strides=1, activation='relu', input_shape=(tokenizer.vocab_size,)))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(len(labels), activation='sigmoid'))

    # specify compilation hyperparameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)])
    
    # Model summmary
    model.summary()

    # fit the model to the training data, monitoring F1 on the dev data
    history = model.fit(
        train_dataset,
        epochs=10,
        batch_size=32,
        validation_data=dev_dataset,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_f1_score",
                mode="max",
                save_best_only=True),
                EarlyStopping(monitor='val_f1_score', mode='max', patience=5, restore_best_weights=True)])

    # Print final validation F1 score
    final_val_f1 = history.history['val_f1_score'][-1]
    print(f"Final Validation F1 Score: {final_val_f1:.4f}")

    # Learning Curves
    plot_learning_curves(history)

def predict(model_path="", input_path="../graduate-project-data/dev.csv"):

    # load the saved model
    model = tf.keras.models.load_model(model_path)

    # load the data for prediction
    # use Pandas here to make assigning labels easier later
    df = pandas.read_csv(input_path)

    # create input features in the same way as in train()
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    #hf_dataset = hf_dataset.map(to_bow) # For Feed Forward NN
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="input_ids", # input_bow for FF
        batch_size=16)

    # generate predictions from model
    predictions = np.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # assign predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = predictions

    # write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(
        method='zip', archive_name=f'submission.csv'))
    
    # Calculate F1 score using tf.keras.metrics.F1Score
    f1_metric = tf.keras.metrics.F1Score(average="micro", threshold=0.5)
    
    # Load the true labels for the dev set
    true_labels_df = pandas.read_csv("../graduate-project-data/dev.csv")
    
    # Calculate the F1 score
    true_labels = true_labels_df.iloc[:, 1:].values
    f1_metric.update_state(true_labels, predictions)
    f1_score = f1_metric.result().numpy()
    
    print(f"F1 Score (Micro): {f1_score}")


def plot_learning_curves(history):

    fig, axs = plt.subplots(2)

    # Create micro f1 subplot
    axs[0].plot(history.history["f1_score"], label="train f1 score")
    axs[0].plot(history.history["val_f1_score"], label="test f1 score")
    axs[0].set_ylabel("F1 Score")
    axs[0].legend(loc="lower right")
    axs[0].set_title("F1 Score eval")

    # Create Loss subplot
    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].plot(history.history["val_loss"], label="test loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss eval")

    plt.show()

    # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')
    # plt.show()

    # Print final validation F1 score
    final_val_f1 = history.history['val_f1_score'][-1]
    print(f"Final Validation F1 Score: {final_val_f1:.4f}")

if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict", "plot_learning_curves"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()
