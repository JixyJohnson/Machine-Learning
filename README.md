Machine Learning Algorithms — Python Implementations

Applied Machine Learning Lab: Core Algorithms in PythonA clean, light, hands-on repository showcasing lightweight Python implementations of foundational Machine Learning, Deep Learning, and Reinforcement Learning algorithms. Built using small datasets and simple configurations, this project is ideal for quick local execution, self-study, and classroom demonstrations.📁 Project StructurePlaintextMachine Learning/
│
├── CatBoost.py
├── decision_tree.py
├── KMeans.py
├── KMeans.png
├── lightGBM.py
├── linear_regression.py
├── QLearning.py
├── Transformer.py
└── XGBoost.py
🧠 Algorithms IncludedProgramAlgorithmParadigmTask / Outputdecision_tree.pyDecision TreeSupervisedMulticlass Classificationlinear_regression.pyLinear RegressionSupervisedContinuous Value RegressionXGBoost.pyXGBoostSupervisedGradient Boosted Trees (Classification)lightGBM.pyLightGBMSupervisedFast Leaf-wise Gradient BoostingCatBoost.pyCatBoostSupervisedSymmetric Tree Gradient BoostingKMeans.pyK-MeansUnsupervisedCentroid-based ClusteringQLearning.pyQ-LearningReinforcementValue-based Policy OptimizationTransformer.pySelf-AttentionDeep LearningScaled Dot-Product Attention Mechanism🔬 Algorithm Breakdown1. Decision Tree (decision_tree.py)Concept: A supervised learning model that recursively partitions the feature space based on feature thresholds.Dataset: Iris Dataset (150 samples, 4 input features, 3 class labels).Split: 70% Training / 30% Testing.Workflow: Iris Dataset ➔ Train-Test Split ➔ Build Model ➔ Fit ➔ Predict ➔ Accuracy EvaluationPythonmodel = DecisionTreeClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
2. Linear Regression (linear_regression.py)Concept: Fits a best-line linear equation ($y = \beta_0 + \beta_1 x$) to model continuous target values.Dataset: Synthetic regression data (make_regression, 100 samples, 1 feature, noise = 15).Metric: Evaluated using Mean Squared Error (MSE). Lower MSE indicates closer predictions.Pythonmodel = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("MSE =", mean_squared_error(y_test, prediction))
3. XGBoost (XGBoost.py)Concept: Extreme Gradient Boosting, an optimized ensemble algorithm using tree-based boosting.Key Hyperparameters:n_estimators=100: Number of boosting trees.learning_rate=0.1: Step size shrinkage used to prevent overfitting.max_depth=3: Maximum depth of individual decision trees.Workflow: Iris Dataset ➔ Train-Test Split ➔ XGBoost Fit ➔ Predict ➔ Accuracy4. LightGBM (lightGBM.py)Concept: A fast, high-performance gradient boosting framework using leaf-wise tree growth.Dataset: Iris Dataset (Evaluated on classification accuracy).Note: Any printed messages regarding "No further splits with positive gain" are expected training warnings and do not affect successful execution.5. CatBoost (CatBoost.py)Concept: Gradient boosting using symmetric trees, designed to handle categorical and numerical features efficiently with minimal tuning.Key Hyperparameters: iterations=100, learning_rate=0.1, depth=3.Note: Configured with verbose=False to maintain clean terminal logs.6. K-Means Clustering (KMeans.py)Concept: An unsupervised algorithm that partitions unlabelled data into $K$ distinct clusters centered around centroids.Dataset: Artificial blob clusters via make_blobs ($N=300$, $K=3$).Workflow: Generate Blobs ➔ Select K=3 ➔ Fit & Predict Cluster Labels ➔ Extract Centroids ➔ Plot VisualizationPythonlabels = model.fit_predict(X)
centroids = model.cluster_centers_
7. Q-Learning (QLearning.py)Concept: A model-free reinforcement learning algorithm where an agent learns optimal actions through interaction and rewards within an environment.Environment:6 States ($0 \rightarrow 1 \rightarrow 2 \rightarrow 3 \rightarrow 4 \rightarrow 5$), where State 5 is the goal.2 Actions: 0 (Move Left), 1 (Move Right).Rewards: $+100$ at the goal, $-1$ per step otherwise.Parameters: Learning rate ($\alpha = 0.8$), Discount factor ($\gamma = 0.9$), trained over 500 episodes.Bellman Update:$$\mathcal{Q}(s,a) \leftarrow \mathcal{Q}(s,a) + \alpha \left[ r + \gamma \max_{a'} \mathcal{Q}(s',a') - \mathcal{Q}(s,a) \right]$$8. Self-Attention / Transformer (Transformer.py)Concept: Demonstrates the core Scaled Dot-Product Attention mechanism that drives modern Transformer architectures.Mechanism: Converts input vectors into Query ($Q$), Key ($K$), and Value ($V$) representations to compute contextual token relationships.$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$Pythonscores = torch.matmul(Q, K.T) / (d_k ** 0.5)
attention = torch.softmax(scores, dim=-1)
output = torch.matmul(attention, V)
⚙️ Prerequisites & SetupEnsure Python is installed on your system. Install the project dependencies via pip:Bashpython -m pip install numpy scikit-learn matplotlib torch xgboost lightgbm catboost
Verify your installation:Bashpython --version
python -m pip --version
▶️ Execution GuideNavigate to the project directory in your terminal or PowerShell prompt:Bashcd "path/to/Machine Learning"
Run any script individually:Bashpython decision_tree.py
python linear_regression.py
python XGBoost.py
python lightGBM.py
python CatBoost.py
python KMeans.py
python QLearning.py
python Transformer.py
💻 Hardware RequirementsLow Resource Overhead: All models rely on small synthetic datasets or lightweight benchmarks.RAM Usage: Runs comfortably on machines with $\le 4\text{ GB}$ RAM.GPU: Not required. The PyTorch attention example runs natively on standard CPU configurations without loading heavy pretrained LLM weights.
📊 Learning Types

The programs demonstrate several major machine learning paradigms.

Supervised Learning

The model learns from input data paired with known target values.

Included:

Decision Tree
Linear Regression
XGBoost
LightGBM
CatBoost
Unsupervised Learning

The model discovers patterns or groups without using target labels.

Included:

K-Means Clustering
Reinforcement Learning

An agent learns by taking actions and receiving rewards or penalties.

Included:

Q-Learning
Deep Learning / Attention

The program demonstrates the self-attention mechanism used in Transformer architectures.

Included:

Transformer / Self-Attention
📊 Quick Comparison MatrixAlgorithmParadigmDataset / EnvironmentTask TypeEvaluation / OutputDecision TreeSupervisedIris DatasetMulticlass ClassificationClassification AccuracyLinear RegressionSupervisedSynthetic (make_regression)Continuous RegressionMean Squared Error (MSE)XGBoostSupervisedIris DatasetEnsemble ClassificationClassification AccuracyLightGBMSupervisedIris DatasetLeaf-wise Gradient BoostingClassification AccuracyCatBoostSupervisedIris DatasetSymmetric Gradient BoostingClassification AccuracyK-MeansUnsupervisedSynthetic (make_blobs)Centroid ClusteringMatplotlib Visual Scatter PlotQ-LearningReinforcement6-State Goal WorldValue/Policy OptimizationFinal Q-Table MatrixTransformerDeep LearningSynthetic Tensor InputsSelf-Attention MechanismContextual Attention Output Matrix🎯 Core Learning ObjectivesThis repository provides clear, minimal implementations to help you master the following practical concepts:Supervised Learning Fundamentals: Understanding classic supervised workflows, including train-test splitting, fitting, and evaluating both classification models (Accuracy) and regression models (MSE).Gradient Boosting Architectures: Comparing key differences in tree-building strategies across leading ensemble frameworks (XGBoost, LightGBM, and CatBoost).Unsupervised Data Partitioning: Discovering underlying patterns and structural clusters without ground-truth targets using distance-based centroids (K-Means).Reinforcement Learning Dynamics: Implementing agent-environment feedback loops, state-action reward functions, and the Bellman equation to optimize policies (Q-Learning).Attention Mechanisms in PyTorch: Implementing Scaled Dot-Product Attention ($\text{softmax}(QK^T / \sqrt{d_k})V$) from scratch without heavy, pretrained model dependencies.Practical Model Execution: Running lightweight ML code efficiently on low-resource environments (under 4 GB RAM / CPU-only configurations).
