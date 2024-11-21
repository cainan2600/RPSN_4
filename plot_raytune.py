from ray import tune
import matplotlib.pyplot as plt
 
if __name__ == '__main__':
    storagePath = "./rayResults/TuneTest"
    tuner = tune.Tuner.restore(path=storagePath)
    res = tuner.get_results()
    bestResult = res.get_best_result(metric="acc", mode="max")
    print(bestResult.config)
    bestResult.metrics_dataframe.plot("training_iteration", "acc")
    plt.show()