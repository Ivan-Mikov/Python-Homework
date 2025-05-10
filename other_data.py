from sklearn import datasets as skd
from knn.realization import KNearestNeighbors, WeightedKNearestNeighbors, accuracy
from knn.preprocessing import train_test_split, get_boxplot_outliers
from vizualize.animationKNN import AnimationKNN
from vizualize.distribution import visualize_distribution
from IPython.display import HTML
import numpy as np


# Для демонстрации работоспособности алгоритма, написана функция
# generate_gender_data(), которая генерирует датасет, в котором
# параметрами точки являются рост и вес, а меткой - пол.
# Причем рост и вес мужчин в среднем выше, чем рост и вес женщин.
# Здесь Женский пол - 0, Мужской пол - 1
# Рост - в см, вес - в кг

def generate_gender_data(num_samples=400, male_ratio=0.5):

    num_males = int(num_samples * male_ratio)
    num_females = num_samples - num_males

    male_height_mean = 175
    male_height_std = 7
    male_weight_mean = 80
    male_weight_std = 10

    female_height_mean = 163
    female_height_std = 6
    female_weight_mean = 65
    female_weight_std = 8

    male_heights = np.random.normal(male_height_mean, male_height_std, num_males)
    male_weights = np.random.normal(male_weight_mean, male_weight_std, num_males)
    male_genders = np.ones(num_males, dtype=int)

    female_heights = np.random.normal(female_height_mean, female_height_std, num_females)
    female_weights = np.random.normal(female_weight_mean, female_weight_std, num_females)
    female_genders = np.zeros(num_females, dtype=int)

    heights = np.concatenate((male_heights, female_heights))
    weights = np.concatenate((male_weights, female_weights))
    genders = np.concatenate((male_genders, female_genders))

    data = np.stack((heights, weights, genders), axis=1)

    np.random.shuffle(data)

    return data  # массив, в котором по столбцам рост, вес и пол


def main():
    data = generate_gender_data()
    points = data[:, :2]
    labels = data[:, 2]

    # 1) Анализ входных данных
    visualize_distribution(
        points,
        ['boxplot', 'violin', 'hist'],
        ['x', 'y'],
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
    anim = animKNN.create_animation(wknn, test_labels, "animation_male_female.gif")
    HTML(anim.to_jshtml())

if __name__ == '__main__':
    main()