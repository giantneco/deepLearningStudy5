import matplotlib.pyplot as plt


def myplot(hist):
    plt.rc('font', family='serif')
    fig = plt.figure()

    num = len(hist.history['val_acc'])

    plt.plot(range(num),
             hist.history['val_acc'],
             label='acc',
             color='black')

    plt.xlim([0,50])
    plt.ylim([0,1.0])

    plt.xlabel('epochs')
    plt.ylabel('validation loss')
    plt.show()

