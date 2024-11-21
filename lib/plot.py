import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('Agg')

def plot_IK_solution(checkpoint_dir, start_epoch, epochs, num_train, num_incorrect_test, num_correct_test):

    draw_epochs = list(range(start_epoch, start_epoch + epochs))

    plt.figure()
    plt.plot(draw_epochs, num_incorrect_test, 'r-', label='Incorrect-No solutions')
    plt.plot(draw_epochs, num_correct_test, 'b-', label='Correct-IK have solutions')

    plt.annotate('{} data sets'.format(num_train), xy=(0.4, 0.5), xycoords='axes fraction', fontsize=12,
                color='gray', horizontalalignment='center', verticalalignment='center')
    # if epoch == 400:
    #     plt.annotate(str(num_correct_test[399]), xy=(draw_epochs[399], num_correct_test[399]),
    #                 xytext=(draw_epochs[399] - 0.1, num_correct_test[399] + 0.8),
    #                 fontsize=8)

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Testing Process ')
    plt.legend()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    file_path = os.path.join(checkpoint_dir, 'Testing Process.png')
    plt.savefig(file_path)

    # plt.show()

def plot_train(checkpoint_dir, start_epoch, epochs, num_train, numError1, numError2, num_incorrect, num_correct):
    draw_epochs = list(range(start_epoch, start_epoch + epochs))
    plt.figure()

    plt.plot(draw_epochs, numError1, 'r-', label='illroot')
    plt.plot(draw_epochs, numError2, 'g-', label='outdom')
    plt.plot(draw_epochs, num_incorrect, 'b-', label='illsolu')
    plt.plot(draw_epochs, num_correct, 'b-', linewidth=3, label='idesolu')

    plt.annotate('{} data sets'.format(num_train), xy=(0.4, 0.5), xycoords='axes fraction', fontsize=12,
                 color='gray', horizontalalignment='center', verticalalignment='center')
    # if epoch == 400:
    #     plt.annotate(str(numNOError2[399]), xy=(draw_epochs[399], numNOError2[399]),
    #                  xytext=(draw_epochs[399] - 0.1, numNOError2[399] + 0.8),
    #                  fontsize=8)

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Process')
    plt.legend()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    file_path = os.path.join(checkpoint_dir, 'Training Process.png')
    plt.savefig(file_path)

    # plt.show()

def plot_train_loss(checkpoint_dir, start_epoch, epochs, echo_loss):
    draw_epochs = list(range(start_epoch, start_epoch + epochs))
    plt.figure()

    plt.plot(draw_epochs, echo_loss, 'r-', label='loss for every epoch')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training loss')
    plt.legend()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    file_path = os.path.join(checkpoint_dir, 'Training loss.png')
    plt.savefig(file_path)

def plot_test_loss(checkpoint_dir, start_epoch, epochs, echo_loss_test):
    draw_epochs = list(range(start_epoch, start_epoch + epochs))
    plt.figure()

    plt.plot(draw_epochs, echo_loss_test, 'r-', label='loss for every epoch')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Test loss')
    plt.legend()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    file_path = os.path.join(checkpoint_dir, 'Test loss.png')
    plt.savefig(file_path)

def plot_2_to_1(checkpoint_dir, start_epoch, epochs, NUM_2_to_1, NUM_mid, NUM_lar):
    draw_epochs = list(range(start_epoch, start_epoch + epochs))
    plt.figure()

    plt.plot(draw_epochs, NUM_2_to_1, 'r-', label='NUM_2_to_1')
    plt.plot(draw_epochs, NUM_mid, 'g-', label='NUM_mid')
    plt.plot(draw_epochs, NUM_lar, 'b-', label='NUM_lar')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('test_2_to_1')
    plt.legend()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    file_path = os.path.join(checkpoint_dir, 'test_2_to_1.png')
    plt.savefig(file_path)

def plot_sametime_solution(checkpoint_dir, start_epoch, epochs, NUM_sametime_solution):
    draw_epochs = list(range(start_epoch, start_epoch + epochs))
    plt.figure()

    plt.plot(draw_epochs, NUM_sametime_solution, 'r-', label='NUM_sametime_solution')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('num_sametime_solution')
    plt.legend()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    file_path = os.path.join(checkpoint_dir, 'sametime_solution.png')
    plt.savefig(file_path)

def plot_no_not_have_solution(checkpoint_dir, start_epoch, epochs, NUM_ALL_HAVE_SOLUTION):
    draw_epochs = list(range(start_epoch, start_epoch + epochs))
    plt.figure()

    plt.plot(draw_epochs, NUM_ALL_HAVE_SOLUTION, 'r-', label='NUM_ALL_HAVE_SOLUTION')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('NUM_ALL_HAVE_SOLUTION')
    plt.legend()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    file_path = os.path.join(checkpoint_dir, 'NUM_ALL_HAVE_SOLUTION.png')
    plt.savefig(file_path)

def plot_no_not_have_solution_test(checkpoint_dir, start_epoch, epochs, NUM_ALL_HAVE_SOLUTION):
    draw_epochs = list(range(start_epoch, start_epoch + epochs))
    plt.figure()

    plt.plot(draw_epochs, NUM_ALL_HAVE_SOLUTION, 'r-', label='NUM_ALL_HAVE_SOLUTION_test')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('NUM_ALL_HAVE_SOLUTION_test')
    plt.legend()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    file_path = os.path.join(checkpoint_dir, 'NUM_ALL_HAVE_SOLUTION_test.png')
    plt.savefig(file_path)