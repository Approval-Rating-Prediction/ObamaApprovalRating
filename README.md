# ObamaApprovalRating
This is a script for predicting President Obama's approval rating. This was an
early project of mine to learn about machine learning algorithms. I selected it
in part because polling has such a rich dataset. Multiple pollsters poll the
president's approval rating on a weekly or even daily basis.

### Machine Learning and data
The script uses a random forest regressor from Python's scikit-learn library.
Data is collected from three sources: the Gallup and Rasmussen daily trackers as
well as the Huffington Post poll collection. The script predicts the next poll
for each source and provides a confidence estimate based on past results.
