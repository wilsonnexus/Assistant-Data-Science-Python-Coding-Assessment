# Author: Wilson Neira
# Take-Home Coding Assessment: L2 Assistant Data Scientist
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset as pandas dataframe
def import_dataset(file_dir):
    # Read CSV file
    dataset = pd.read_csv(file_dir)
    # Remove leading spaces from column labels
    dataset.columns = dataset.columns.str.strip()
    return dataset

# Answers for questions from 1. Regression Modelling
def regression_modeling():
    dataset = import_dataset("Iris_Data.csv")

    ## 1. a) How many irises belong to each species?
    species_count = dataset["Labels"].value_counts()
    print("1. a)", species_count[0], "irises belong to species 0,",
          species_count[1], "irises belong to species 1, and",
          species_count[2], "irises belong to species 2.\n")

    ## 1. b) Make a scatterplot of petal length vs sepal length. Color the dots according to species.
    ##    Document your observations (2-3 sentences)
    # Create a new figure
    fig, ax = plt.subplots()
    # Set the title and axis labels
    ax.set_title('Petal Length Vs Sepal Length')
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Petal Length ')
    # Add a label below the scatter plot
    plt.text(0, -0.1, '1. b)', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

    # Define a color map for the three species
    colors = ['r', 'g', 'b']
    # Group the data by species and plot each group with a different color
    for i, (species, group) in enumerate(dataset.groupby('Labels')):
        plt.scatter(group["Sepal Length"], group["Petal Length"],
                    c=colors[i], label=dataset["Labels"])

    ax.legend(["Species 0", "Species 1", "Species 2"])
    # Observations
    print("1. b) Species 0 had about the smallest, "
          "species 1 had the second smallest, and "
          "species 2 had the largest\nsepal and petal length. "
          "There is a sepal and petal length positive relationship "
          "for species 1 and 2.\nThere is a not so strong "
          " sepal and petal length relationship for "
          "when compared to species 1 and 2.\n")

    ## 1. c) Fit a regression model predicting sepal length based on petal length, petal width
    ##    and sepal width (you do not need to test any of the regression assumptions).
    # Fit a linear regression model
    lm = LinearRegression().fit(dataset[["Petal Length", "Petal Width", "Sepal Width"]], dataset["Sepal Length"])

    ## 1. d) Describe the results of your regression, focusing on the relationship between
    ##    sepal length and petal length.
    # Print the model coefficients
    print(f'1. d) Intercept: {lm.intercept_:.2f}')
    print(f'Coefficients: {lm.coef_}')
    # Print the R-squared value
    print('R-squared:', lm.score(dataset[["Petal Length", "Petal Width", "Sepal Width"]], dataset["Sepal Length"]))
    print("1. d) My regression model has 3 coefficients which corresponds to the given predictors. "
          "The first \ncoefficient", lm.coef_[0], "represents the change in petal length "
          "for a one-unit increase in \nsepal length, holding all other predictor variables "
          "constant. Shown by the coefficient we can \nsee that the relationship between "
          "the sepal length and petal length is positive, so when sepal \nlength increases,"
          " petal length also increases. By the regression model R-squared value we notice \n"
          "that there exists a strong relationship between the predictors and petal length.\n")

    ## 1. e) Extra Credit: Fit a regression model predicting sepal length based on petal length,
    ##    petal width, sepal width and species (you do not need to test for any of the
    ##    “classical” regression assumptions).  This is the same as part c but also with
    ##    species as a predictor. Describe the results.
    lm = LinearRegression().fit(dataset[["Petal Length", "Petal Width", "Sepal Width", "Labels"]],
         dataset["Sepal Length"])
    print(f'1. e) Intercept: {lm.intercept_:.2f}')
    print(f'Coefficients: {lm.coef_}')
    print('R-squared:', lm.score(dataset[["Petal Length", "Petal Width", "Sepal Width", "Labels"]],
          dataset["Sepal Length"]))
    print("1. e) My regression model has 4 coefficients which corresponds to the given predictors. "
          "The first \ncoefficient", lm.coef_[0], "represents the change in petal length "
          "for a one-unit increase in \nsepal length, holding all other predictor variables "
          "constant. Shown by the coefficient we can \nsee that the relationship between "
          "the sepal length and petal length is positive and greater \nthan part d, so when sepal "
          "length increases, petal length also increases. By the regression \nmodel R-squared value",
          lm.score(dataset[["Petal Length", "Petal Width", "Sepal Width", "Labels"]],
          dataset["Sepal Length"]), "we notice that it is greater than part d and there \nexists "
          "a strong relationship between the predictors and petal length.\n")


# Answers for questions from 2. Implementing an Edit-Distance Algorithm
def edit_distance_algorithm():
    ## 2. 1) Add .5 to the Hamming distance if a capital letter is switched for a
    ##    lower case letter unless it is in the first position.  Examples include:

    print("2. 1) a. \"Kitten\" and \"kitten\" have a distance of", hamming_distance_mod("Kitten", "kitten"))
    print("2. 1) b. \"kitten\" and \"KitTen\" have a  Hamming distance of", hamming_distance_mod("kitten", "KitTen"), ".")
    print("2. 1) c. \"Puppy\" and \"POppy\" have a distance of", hamming_distance_mod("Puppy", "POppy"),
          "(1 for the different \nletter, additional .5 for the different capitalization).\n")

    ## 2. 2) Consider S and Z (and s and z) to be the same letter. For example,
    ##    "analyze" has a distance of 0 from "analyse".
    print("Test Outcome", hamming_distance_mod("make", "Mage"), "Expected 1")
    print("Test Outcome", hamming_distance_mod("maisy", "MaiZy"), "Expected 0.5")
    print("Test Outcome", hamming_distance_mod("Eagle", "Eager"), "Expected 2")
    print("Test Outcome", hamming_distance_mod("Sentences work too", "Sentences wAke too"), "Expected 3.5\n")
    ##    Use the program you wrote to score the following strings:
    print("2. 2) a) \"data Science\" to \"Data Sciency\" have a distance of", hamming_distance_mod("data Science", "Data Sciency"))
    print("2. 2) b) \"organizing\" to \"orGanising\" have a  Hamming distance of", hamming_distance_mod("organizing", "orGanising"))
    print("2. 2) c) \"AGPRklafsdyweIllIIgEnXuTggzF\" to \"AgpRkliFZdiweIllIIgENXUTygSF\" have a distance of",
          hamming_distance_mod("AGPRklafsdyweIllIIgEnXuTggzF", "AgpRkliFZdiweIllIIgENXUTygSF"))
    ##    Describe a scenario (3-4 sentences) where implementing the standard
    ##    Hamming distance algorithm would be applicable.
    print("\n2. 2) a) The Hamming distance algorithm would be applicable in social policy,"
          " \nthe standard Hamming distance algorithm can be applied to analyze "
          "social \nmedia data to identify potential hate speech or discriminatory "
          "language. \nBy representing social media posts or comments as binary strings "
          "or using \none-hot encoding, the Hamming distance can be used to compare posts "
          "with \nknown patterns of hate speech and identify potential instances of "
          "hate \nspeech or discrimination. This can help policymakers to develop targeted"
          " \ninterventions to reduce hate speech and promote inclusivity. For example, "
          "\nthe Hamming distance can be used to compare social media posts with known "
          "\npatterns of hate speech and identify potential patterns of discriminatory "
          "\nlanguage that may require further investigation or intervention.\n")


def hamming_distance_mod(string1: str, string2: str) -> int:
    """Return the Hamming distance between two strings."""
    if len(string1) != len(string2):
        raise ValueError("Strings must be of equal length.")
    dist_counter = 0
    for n in range(len(string1)):
        letter1 = string1[n]
        letter2 = string2[n]
        # Set s or S equal to z or Z respectively
        if letter1 == 'z':
            letter1 = 's'
        if letter1 == 'Z':
            letter1 = 'S'
        if letter2 == 'z':
            letter2 = 's'
        if letter2 == 'Z':
            letter2 = 'S'
        if letter1 != letter2 and n != 0:
            # Check if both letters are the same when upper-cased
            if letter1.upper() != letter2.upper():
                dist_counter += 1
            # Check if both letters have the same ASCII value
            if ord(letter1) != ord(letter2):
                if letter1.isupper() or letter2.isupper():
                    dist_counter += 0.5
    return dist_counter

# Answers for questions from 3. Data Cleaning
def data_cleaning():
    ## 3. a) How many of the field descriptions reference a perspective that is
    ##    not standard (i.e. viewed from the top, bottom, front or rear)?
    ##    Specifically, write code to count how many of the rows have the words
    ##    "view" or "perspective" but do not include "bottom", "top", "front"
    ##    or "rear" in  the text field?
    dataset = import_dataset("patent_drawing data.csv")
    count = 0
    for text in dataset["text"]:
        word_list = []
        for word in text.split():
            # check if the word contains only letters
            if word.isalpha():
                word_list.append(word)
        not_include = True
        include = False
        for word in word_list:
            if (word.upper() == "BOTTOM" or word.upper() == "TOP" or
                    word.upper() == "FRONT" or word.upper() == "REAR"):
                not_include = False
            if not_include:
                if word.upper() == "PERSPECTIVE" or word.upper() == "VIEW":
                    include = True
        if include:
            count += 1

    print("3. a) The number of field descriptions reference a "
          "perspective that is not standard is", count)

    ## 3. b) What is the average number of drawing descriptions per patent?
    unique_patents = dataset["patent_id"].unique()
    n = len(unique_patents)
    total = 0
    for patent in unique_patents:
        # select rows where column "patent_id" contains the patent value
        dataset_patent = dataset[dataset["patent_id"] == patent]
        total += len(dataset_patent)
    average = total/n

    print("3. b) The average number of drawing descriptions per patent is", average)


if __name__ == '__main__':
    # Answers for questions 1. Regression Modelling
    regression_modeling()
    # Answers for questions from 2. Implementing an Edit-Distance Algorithm
    edit_distance_algorithm()
    # Answers for questions from 3. Data Cleaning
    data_cleaning()
    # Plot the scatterplot
    plt.show()