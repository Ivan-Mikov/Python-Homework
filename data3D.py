from sklearn.datasets import make_classification
from knn.realization import KNearestNeighbors, WeightedKNearestNeighbors, accuracy
from knn.preprocessing import train_test_split, get_boxplot_outliers
from vizualize.animationKNN import AnimationKNN
from vizualize.distribution import visualize_distribution
from IPython.display import HTML


def main():
    points, labels = make_classification(
        n_samples=100,
        random_state=42,
        n_features=3,
        n_classes=2,
        n_informative=3,
        n_redundant=0
        )

    # 1) Анализ входных данных
    visualize_distribution(
        points,
        ['boxplot', 'violin', 'hist'],
        ['x', 'y', 'z'],
        "distribution.png"
    )

    # 2) Предобработка
    # 2.1) Исключение лишнего
    outliers_indices = get_boxplot_outliers(points)
    print(f"Количество выбросов: {len(outliers_indices)}")
    print(f"Индексы выбросов: {outliers_indices}")

    # 2.2) Подготовка данных
    train_points, train_labels, test_points, test_labels = train_test_split(points, labels)

    # 3) Алгоритмы
    # 3.1) Классический KNN
    knn = KNearestNeighbors()
    knn.fit(train_points, train_labels)
    predict_labels_knn = knn.predict(test_points)

    # 3.2) Взвешенный KNN
    wknn = WeightedKNearestNeighbors()
    wknn.fit(train_points, train_labels)
    predict_labels_wknn = wknn.predict(test_points)

    # 4) Оценка качества
    knn_quality = accuracy(test_labels, predict_labels_knn)
    wknn_quality = accuracy(test_labels, predict_labels_wknn)
    print(f"Процент попаданий KNN: {knn_quality * 100} %")
    print(f"Процент попаданий взвешенного KNN: {wknn_quality * 100} %")

    # 5) Визуализация результатов
    animKNN = AnimationKNN(test_points)
    anim = animKNN.create_animation(wknn, test_labels, "animationKNN.gif")
    HTML(anim.to_jshtml())

if __name__ == '__main__':
    main()