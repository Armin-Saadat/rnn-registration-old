import neurite as ne


def visualize(image, label):
    print(image.shape)
    pics = [image, label]
    titles = ['image', 'label']
    ne.plot.slices(pics, titles=titles, cmaps=['gray'], do_colorbars=True, imshow_args=[{'origin': 'lower'}]);
