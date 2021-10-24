import neurite as ne


def visualize(images, labels, slice_idx):
    image = images.squeeze()[slice_idx].detach().cpu().numpy()
    label = labels.squeeze()[slice_idx].detach().cpu().numpy()
    pics = [image, label]
    titles = ['image', 'label']
    ne.plot.slices(pics, titles=titles, cmaps=['gray'], do_colorbars=True, imshow_args=[{'origin': 'lower'}]);
