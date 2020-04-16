import pandas as pd
from process import generate

def data_provider(name):

    if name == 'code':
        post = pd.read_csv('./data/code/post.csv')
        comment = pd.read_csv('./data/code/comment.csv')
        thread = pd.read_csv('./data/code/thread.csv')
        threads, labels, features = generate(thread, post, comment)

    elif name == 'ani':
        post_1 = pd.read_csv('./data/ani_1/post.csv')
        comment_1 = pd.read_csv('./data/ani_1/comment.csv')
        thread_1 = pd.read_csv('./data/ani_1/thread.csv')

        threads_1, labels_1, features_1 = generate(thread_1, post_1, comment_1)

        post_2 = pd.read_csv('./data/ani_2/post.csv')
        comment_2 = pd.read_csv('./data/ani_2/comment.csv')
        thread_2 = pd.read_csv('./data/ani_2/thread.csv')

        threads_2, labels_2, features_2 = generate(thread_2, post_2, comment_2)

        threads = threads_1 + threads_2
        labels = labels_1 + labels_2
        features = pd.concat([features_1, features_2], axis=0)

    elif name == 'astro':
        post_1 = pd.read_csv('./data/astro_1/post.csv')
        comment_1 = pd.read_csv('./data/astro_1/comment.csv')
        thread_1 = pd.read_csv('./data/astro_1/thread.csv')

        post_2 = pd.read_csv('./data/astro_2/post.csv')
        comment_2 = pd.read_csv('./data/astro_2/comment.csv')
        thread_2 = pd.read_csv('./data/astro_2/thread.csv')

        threads_1, labels_1, features_1 = generate(thread_1, post_1, comment_1)
        threads_2, labels_2, features_2 = generate(thread_2, post_2, comment_2)

        threads = threads_1 + threads_2
        labels = labels_1 + labels_2
        features = pd.concat([features_1, features_2], axis=0)

    elif name == 'chi':
        post = pd.read_csv('./data/chi/post.csv')
        comment = pd.read_csv('./data/chi/comment.csv')
        thread = pd.read_csv('./data/chi/thread.csv')

        threads, labels, features = generate(thread, post, comment)

    elif name == 'cli':
        post_1 = pd.read_csv('./data/cli_1/post.csv')
        comment_1 = pd.read_csv('./data/cli_1/comment.csv')
        thread_1 = pd.read_csv('./data/cli_1/thread.csv')

        post_2 = pd.read_csv('./data/cli_2/post.csv')
        comment_2 = pd.read_csv('./data/cli_2/comment.csv')
        thread_2 = pd.read_csv('./data/cli_2/thread.csv')

        threads_1, labels_1, features_1 = generate(thread_1, post_1, comment_1)
        threads_2, labels_2, features_2 = generate(thread_2, post_2, comment_2)

        threads = threads_1 + threads_2
        labels = labels_1 + labels_2
        features = pd.concat([features_1, features_2], axis=0)

    elif name == 'cri':
        threads = []
        labels = []
        features = None

        for i in range(1, 4):
            post = pd.read_csv('./data/cri_' + str(i) + '/post.csv')
            comment = pd.read_csv('./data/cri_' + str(i) + '/comment.csv')
            thread = pd.read_csv('./data/cri_' + str(i) + '/thread.csv')

            threads_1, labels_1, features_1 = generate(thread, post, comment)
            threads = threads + threads_1
            labels = labels + labels_1
            if i == 1:
                features = features_1
            else:
                features = pd.concat([features, features_1], axis=0)


    elif name == 'edi':
        threads = []
        labels = []
        features = None
        for i in range(1, 3):
            post = pd.read_csv('./data/edi_' + str(i) + '/post.csv')
            comment = pd.read_csv('./data/edi_' + str(i) + '/comment.csv')
            thread = pd.read_csv('./data/edi_' + str(i) + '/thread.csv')
            threads_1, labels_1, features_1 = generate(thread, post, comment)
            threads = threads + threads_1
            labels = labels + labels_1
            if i == 1:
                features = features_1
            else:
                features = pd.concat([features, features_1], axis=0)

    elif name == 'music':
        post = pd.read_csv('./data/music/post.csv')
        comment = pd.read_csv('./data/music/comment.csv')
        thread = pd.read_csv('./data/music/thread.csv')
        threads, labels, features = generate(thread, post, comment)

    elif name == 'phi':
        threads = []
        labels = []
        features = None
        for i in range(1, 5):
            post = pd.read_csv('./data/phi_' + str(i) + '/post.csv')
            comment = pd.read_csv('./data/phi_' + str(i) + '/comment.csv')
            thread = pd.read_csv('./data/phi_' + str(i) + '/thread.csv')

            threads_1, labels_1, features_1 = generate(thread, post, comment)
            threads = threads + threads_1
            labels = labels + labels_1
            if i == 1:
                features = features_1
            else:
                features = pd.concat([features, features_1], axis=0)

    return threads, labels, features