import re
import pickle
import pandas as pd
import numpy as np
from arg_extractor import get_args
args = get_args()
def tokenize_and_clean(string, tokenizer):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # for math equations
    string = re.sub(r"https?\:\/\/[a-zA-Z0-9][a-zA-Z0-9\.\_\?\=\/\%\-\~\&]+", " ", string)
    string = re.sub(r'^https?:\/\/.*[\r\n]*', '', string)
    string = re.sub('<[^>]+>', ' ', string)

    string = re.sub(r"\$\$.*?\$\$", " ", string)
    string = re.sub(r"\(.*\(.*?\=.*?\)\)", " ", string)
    string = re.sub(r"\\\(\\mathop.*?\\\)", " ", string)
    string = re.sub(r"\\\[\\mathop.*?\\\]", " ", string)
    string = re.sub(r"[A-Za-z]+\(.*?\)", " ", string)
    string = re.sub(r"[A-Za-z]+\[.*?\]", " ", string)
    string = re.sub(r"[0-9][\+\*\\\/\~][0-9]", " ", string)
    string = re.sub(r"<MATH>\s*[\+\-\*\\\/\~][0-9]", " ", string)

    string = re.sub(r"<MATH>\s*[\+\-\*\\\/\~\=]", " ", string)
    string = re.sub(r"[\+\-\*\\\/\~\=]\s*<MATH>", " ", string)

    string = re.sub(r"[\+\*\\\/\~]", " ", string)
    string = re.sub(r"(<MATH>\s*)+", " ", string)

    # for time
    string = re.sub(r"[0-9][0-9]?:[0-9][0-9]?", "", string)

    # for url's

    # for english sentences
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = string.strip().lower()

    deleted_words = ['br', 'href', '\\', 'li', 'div', 'br', 'span', 'p']


    querywords = string.split()

    resultwords = [word for word in querywords if word.lower() not in deleted_words]
    string = ' '.join(resultwords)

    # tokenize string, filter for stopwords and return
    encoded_sent = tokenizer.encode(
        string,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    )

    return encoded_sent


def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_sequences(posts, tokenizer):
    '''
    turns words in pieces of text into padded
    sequences of word indices correspodning to
    the vocab
    '''

    post_tkns = []
    for post in posts:
        post_tkns.append(tokenize_and_clean(str(post), tokenizer))

    # get the length of each sentence
    post_lengths = [len(text) for text in post_tkns]

    # create an empty matrix with padding tokens
    if len(post_lengths) == 0:
        padded_posts = np.zeros((1, 512), dtype=int)
        padded_posts[0,0] = 101
        padded_posts[0,511] = 102
    else:
        if max(post_lengths) <=512:
            max_post_length = max(post_lengths)
        else:
             max_post_length = 512
        thread_length = len(posts)
        padded_posts = np.zeros((thread_length, max_post_length), dtype=int)

        # copy over the actual sequences
        for i, post_length in enumerate(post_lengths):
            if post_length == 0:
                continue
            else:
                if post_length <= max_post_length:
                    padded_posts[i, :post_length] = np.array(post_tkns[i])
                else:
                    padded_posts[i, :] = np.array(post_tkns[i][:max_post_length-1] + [102])

    return padded_posts


def generate(thread_data_useful, post_data, comment_data):
    threads = []

    for idx in thread_data_useful['id'].values:
        thread = thread_data_useful.loc[thread_data_useful['id'] == idx]
        if thread['instructor_replied'].values == 0:
            posts = post_data.loc[post_data['thread_id'] == idx].sort_values(by=['post_time'])[
                'post_text'].values.tolist()
            comments = comment_data.loc[comment_data['thread_id'] == idx].sort_values(by=['post_time'])[
                'comment_text'].values.tolist()
        else:
            posts = post_data.loc[post_data['thread_id'] == idx]
            comments = comment_data.loc[comment_data['thread_id'] == idx]
            time_list = list(posts.loc[posts['forum_title'] != 'Student']['post_time']) + list(
                comments.loc[comments['forum_title'] != 'Student']['post_time'])
            time_list.sort()
            posts = posts.loc[posts['post_time'] < time_list[0]].sort_values(by=['post_time'])[
                'post_text'].values.tolist()
            comments = comments.loc[comments['post_time'] < time_list[0]].sort_values(by=['post_time'])[
                'comment_text'].values.tolist()

        threads.append(posts + comments)

    labels = thread_data_useful['instructor_replied'].values.tolist()

    col_names = ['forum_name', 'thread_id', 'thread_ano', 'if_student',
                 'approved', 'unresolved', 'deleted', 'start_time',
                 'end_time', 'num_posts', 'num_comments', 'num_views', 'votes', 'if_lecture',
                 'if_cw', 'sum_votes', 'mean_time', 'var_time', 'sum_assess',
                 'sum_problems', 'sum_thanks', 'sum_requests', 'sum_transition', 'thread_txt', 'intervene']
    code_thread_X = pd.DataFrame(columns=col_names)
    create(thread_data_useful, post_data, comment_data, code_thread_X, 'code')
    return threads, labels, code_thread_X


def create(thread_data_useful, post_data, comment_data, thread_X, name):
    for row in range(thread_data_useful.shape[0]):
        current_thread = thread_data_useful.iloc[row]

        # Extract thread_level information
        intervene = current_thread['instructor_replied']
        forum_name = name
        thread_id = current_thread['id']
        thread_ano = current_thread['anonymous']
        if_student = int(current_thread['forum_title'] == 'Student')
        approved = current_thread['approved']
        unresolved = current_thread['unresolved']
        deleted = current_thread['deleted']
        start_time = current_thread['posted_time']
        end_time = current_thread['last_updated_time']

        num_views = current_thread['num_views']
        votes = current_thread['votes']

        thread_title = current_thread['title'].lower()
        if ('lecture' in thread_title) or ('lectures' in thread_title) or ('video' in thread_title) or (
                'videos' in thread_title):
            if_lecture = 1
        else:
            if_lecture = 0

        if (('assignment' in thread_title) or ('assignments' in thread_title)
                or ('quiz' in thread_title) or ('quizzes' in thread_title)
                or ('grade' in thread_title) or ('grades' in thread_title)
                or ('project' in thread_title) or ('projects' in thread_title)
                or ('exam' in thread_title) or ('exams' in thread_title)):
            if_cw = 1
        else:
            if_cw = 0

        sum_votes = 0
        time_list = []
        mean_time = 0
        var_time = 0
        num_posts = 0
        num_comments = 0
        sum_assess = 0
        sum_problems = 0
        sum_thanks = 0
        sum_requests = 0
        sum_transition = 0

        thread_txt = ''

        # Extract information from posts and comments
        posts_df = post_data.loc[post_data['thread_id'] == thread_id]
        comments_df = comment_data.loc[comment_data['thread_id'] == thread_id]
        len_posts = len(posts_df)

        # Make sure there are posts in the thread
        if len_posts == 0:
            continue

        assess_list = ['grade', 'grades', 'exam', 'exams', 'assignment', 'assignments', 'quiz', 'quizzes',
                       'reading', 'readings', 'project', 'projects']
        tech_list = ['problem', 'problems', 'error', 'errors', 'mistake', 'mistakes']
        conclusive_list = ['thank', 'thanks', 'appreciate']
        request_list = ['request', 'requests', 'submit', 'submits', 'suggest', 'suggests']
        transition_list = ['but', 'however', 'nevertheless', 'nonetheless', 'despite', 'in spite of', 'although']

        # If there is no intervention in this thread, count everything
        if intervene == 0:

            time_list = time_list + list(posts_df['post_time'])
            time_list = time_list + list(comments_df['post_time'])

            if max(time_list) - min(time_list) != 0:
                time_list = [float(i - min(time_list)) / (max(time_list) - min(time_list)) for i in time_list]
            else:
                time_list = [1]

            mean_time = np.mean(time_list)
            var_time = np.var(time_list)

            sum_votes = sum(list(posts_df['votes'])) + sum(list(comments_df['votes']))

            text = list(posts_df['post_text']) + list(comments_df['comment_text'])

            thread_txt = clean(' '.join(str(v) for v in text))


            num_posts = len(list(posts_df['post_text']))
            num_comments = len(list(comments_df['comment_text']))



            for s in (list(posts_df['post_text']) + list(comments_df['comment_text'])):
                s = str(s)
                for word in assess_list:

                    sum_assess += s.count(word)



                for word in tech_list:
                    sum_problems += s.count(word)

                for word in conclusive_list:
                    sum_thanks += s.count(word)

                for word in request_list:
                    sum_requests += s.count(word)

                for word in transition_list:
                    sum_transition += s.count(word)

        # If there is an intervention in this thread, curtail the rest of thread
        else:

            post_time_list = list(posts_df.loc[posts_df['forum_title'] != 'Student']['post_time']) + list(
                comments_df.loc[comments_df['forum_title'] != 'Student']['post_time'])

            post_time_list.sort()

            posts_df_new = None
            comments_df_new = None
            for n in range(len(post_time_list)):
                mini = post_time_list[n]
                posts_df_new = posts_df.loc[posts_df['post_time'] < mini]
                comments_df_new = comments_df.loc[comments_df['post_time'] < mini]
                if ((len(posts_df_new) + len(comments_df_new)) == 0):
                    continue
                else:
                    break

            time_list = time_list + list(posts_df_new['post_time'])
            time_list = time_list + list(comments_df_new['post_time'])
            if max(time_list) - min(time_list) != 0:
                time_list = [float(i - min(time_list)) / (max(time_list) - min(time_list)) for i in time_list]
            else:
                time_list = [1]

            mean_time = np.mean(time_list)
            var_time = np.var(time_list)

            sum_votes = sum(list(posts_df_new['votes'])) + sum(list(posts_df_new['votes']))

            text = list(posts_df_new['post_text']) + list(comments_df_new['comment_text'])
            num_posts = len(text)
            thread_txt = clean(' '.join(str(v) for v in text))

            num_posts = len(list(posts_df_new['post_text']))
            num_comments = len(list(comments_df_new['comment_text']))

            for s in (list(posts_df_new['post_text']) + list(comments_df_new['comment_text'])):
                s = str(s)
                for word in assess_list:
                    sum_assess += s.count(word)

                for word in tech_list:
                    sum_problems += s.count(word)

                for word in conclusive_list:
                    sum_thanks += s.count(word)

                for word in request_list:
                    sum_requests += s.count(word)

                for word in transition_list:
                    sum_transition += s.count(word)

        thread_X.loc[len(thread_X)] = [forum_name, thread_id, thread_ano, if_student,
                                       approved, unresolved, deleted, start_time,
                                       end_time, num_posts, num_comments, num_views, votes, if_lecture,
                                       if_cw, sum_votes, mean_time, var_time, sum_assess,
                                       sum_problems, sum_thanks, sum_requests, sum_transition, thread_txt, intervene]


def clean(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # for math equations
    string = re.sub(r"https?\:\/\/[a-zA-Z0-9][a-zA-Z0-9\.\_\?\=\/\%\-\~\&]+", " ", string)
    string = re.sub(r'^https?:\/\/.*[\r\n]*', '', string)

    string = re.sub(r"\$\$.*?\$\$", " ", string)
    string = re.sub(r"\(.*\(.*?\=.*?\)\)", " ", string)
    string = re.sub(r"\\\(\\mathop.*?\\\)", " ", string)
    string = re.sub(r"\\\[\\mathop.*?\\\]", " ", string)
    string = re.sub(r"[A-Za-z]+\(.*?\)", " ", string)
    string = re.sub(r"[A-Za-z]+\[.*?\]", " ", string)
    string = re.sub(r"[0-9][\+\*\\\/\~][0-9]", " ", string)
    string = re.sub(r"<MATH>\s*[\+\-\*\\\/\~][0-9]", " ", string)

    string = re.sub(r"<MATH>\s*[\+\-\*\\\/\~\=]", " ", string)
    string = re.sub(r"[\+\-\*\\\/\~\=]\s*<MATH>", " ", string)

    string = re.sub(r"[\+\*\\\/\~]", " ", string)
    string = re.sub(r"(<MATH>\s*)+", " ", string)

    # for time
    string = re.sub(r"[0-9][0-9]?:[0-9][0-9]?", "", string)

    # for url's

    # for english sentences
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = string.strip().lower()

    return string
