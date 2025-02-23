
# -*- coding: utf-8 -*-
"""
Created on  Feb 13 2025

@author: sac
"""


import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt


from util_sac.pandas.print_df import print_partial_markdown
from util_sac.data.print_array_info import print_array_info
from util_sac.image_processing.reduce_palette import reduce_palette

import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def objective(trial, X, y):
	"""
	Objective function for optimizing Decision Tree parameters using Optuna.

	Parameters
	----------
	trial : feature_analysis.trial.Trial
		Optuna에서 제공하는 trial 객체로, 하이퍼파라미터 샘플링에 사용됩니다.
	X : array-like of shape (n_samples, n_features)
		모델에 입력될 feature 데이터입니다.
	y : array-like of shape (n_samples,)
		모델의 타깃 레이블로, 정수 레이블 형태의 1차원 array여야 합니다.

	Returns
	-------
	float
		Cross-validation 결과의 평균 오차(또는 정확도에 -1을 곱한 값 등)를 반환합니다.
		Optuna에서 이 값을 기준으로 최적화가 진행됩니다.
	"""
	# Decision Tree 파라미터 검색 공간 정의
	max_depth = trial.suggest_int('max_depth', 1, 10)
	criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
	min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
	min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)


	# Decision Tree 모델 생성
	model = DecisionTreeClassifier(
		max_depth=max_depth,
		criterion=criterion,
		min_samples_split=min_samples_split,
		min_samples_leaf=min_samples_leaf,
	)

	# 'f1_macro'로 설정하면 각 클래스별 F1 점수를 구하고, 평균(macro 평균)을 산출
	scores = cross_val_score(model, X, y, cv=3, scoring='f1_macro')
	return -scores.mean()  # Optuna는 최소화를 수행하므로 음수 부호를 붙여줍니다.


def run_optimize_tree(X, y):
	"""
	Decision Tree 하이퍼파라미터를 Optuna로 최적화한 뒤,
	최적의 하이퍼파라미터로 학습된 모델을 반환합니다.

	Parameters
	----------
	X : array-like of shape (n_samples, n_features)
		모델에 입력될 feature 데이터입니다.
	y : array-like of shape (n_samples,)
		모델의 타깃 레이블로, 정수 레이블 형태의 1차원 array여야 합니다.

	Returns
	-------
	model : DecisionTreeClassifier
		최적의 하이퍼파라미터로 학습된 Decision Tree 모델을 반환합니다.
	"""
	study = optuna.create_study(direction='minimize')
	study.optimize(lambda trial: objective(trial, X, y), n_trials=100)

	best_params = study.best_params
	print("Best hyperparameters:", best_params)
	print("Best score:", -study.best_value)

	# 최적 하이퍼파라미터로 모델 생성 후 전체 train 데이터로 학습
	model = DecisionTreeClassifier(**best_params)
	model.fit(X, y)
	return model


def feature_importance_tree(X_train, y_train, X_test, y_test, feature_names):
	"""
	run_optimize_tree로 구한 최적화된 Decision Tree 모델을 이용하여
	test set에 대한 accuracy, confusion matrix, macro-F1 score를 출력하고,
	feature importance를 시각화합니다.

	Parameters
	----------
	X_train : array-like of shape (n_samples, n_features)
		학습에 사용되는 feature 데이터입니다.
	y_train : array-like of shape (n_samples,)
		학습에 사용되는 타깃 레이블입니다.
	X_test : array-like of shape (n_samples, n_features)
		평가에 사용되는 feature 데이터입니다.
	y_test : array-like of shape (n_samples,)
		평가에 사용되는 타깃 레이블입니다.

	Returns
	-------
	model : DecisionTreeClassifier
		최적의 하이퍼파라미터로 학습된 모델을 반환합니다.
	"""
	from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

	# run_optimize_tree 함수를 이용해 최적화된 모델 학습
	model = run_optimize_tree(X_train, y_train)

	# 예측 및 평가
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	cm = confusion_matrix(y_test, y_pred)
	macro_f1 = f1_score(y_test, y_pred, average='macro')

	print("Test accuracy:", accuracy)
	print("Confusion matrix:\n", cm)
	print("Macro-F1 score:", macro_f1)

	# feature importance 출력 및 시각화
	importances = model.feature_importances_

	# 시각화
	plt.close()
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))

	# 첫 번째 subplot: Feature Importance
	axes[0].barh(feature_names, importances, color='skyblue')
	axes[0].set_xlabel('Importance')
	axes[0].set_title('Feature Importances (Random Forest)')
	axes[0].invert_yaxis()

	# 두 번째 subplot: Confusion Matrix (matplotlib imshow 사용)
	im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	axes[1].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}, Macro-F1: {macro_f1:.2f}')
	axes[1].set_xlabel('Predicted Label')
	axes[1].set_ylabel('True Label')

	# Colorbar 추가
	fig.colorbar(im, ax=axes[1])

	# 각 셀에 숫자 표기 (값에 따라 글자색 변경)
	thresh = cm.max() / 2.0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			color = "white" if cm[i, j] > thresh else "black"
			axes[1].text(j, i, format(cm[i, j], 'd'),
						 ha="center", va="center", color=color)

	plt.tight_layout()
	return model, fig


if __name__ == '__main__':

	df_feature = pd.DataFrame(features)
	X = df_feature.values
	y = np.array(y)
	feature_names = df_feature.columns
	"""
	X          NumPy Array          (185, 15)                   21.68 KB float64
	y          NumPy Array          (185,)                       1.45 KB int64
	"""

	from sklearn.model_selection import train_test_split

	# train / test 분할
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.3,
		random_state=42
	)

	feature_importance_tree(X_train, y_train, X_test, y_test, feature_names)

