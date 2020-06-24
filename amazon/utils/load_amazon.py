import numpy as np

def load_amazon(filepath = './data/amazon_tf-idf.npz', domains=['books', 'dvd', 'electronics', 'kitchen_&_housewares']):
    f = np.load(filepath)
    amazon_x = f['X']
    amazon_y = f['y']
    amazon_domains = f['domains']
    domain_list = np.unique(amazon_domains)
    if domains is None:
        domains = domain_list
    X_amazon = []
    y_amazon = []
    domain_keep = []
    for i in range(len(domain_list)):
        d = domain_list[i]
        if d in domains:
            idx = np.where(amazon_domains==d)[0]
            X_amazon.append(amazon_x[idx])
            y_amazon.append(amazon_y[idx])
            domain_keep.append(d)
    return X_amazon, y_amazon, domain_keep