from word_vectors import distinct_words, compute_co_occurrence_matrix, reduce_to_k_dim
import numpy as np

def test_distinct_words():
    test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]
    test_corpus_words, num_corpus_words = distinct_words(test_corpus)

    ans_test_corpus_words = sorted(list(set(["START", "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", "END"])))
    ans_num_corpus_words = len(ans_test_corpus_words)

    # Test correct number of words
    assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)

    # Test correct words
    assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))

    # Print Success
    print ("-" * 80)
    print("Passed All Tests!")
    print ("-" * 80)

def test_co_occurance_matrix():
    test_corpus = ["START All that glitters isn't gold END".split(" "),
                   "START All's well that ends well END".split(" ")]
    M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)

    # Correct M and word2Ind
    M_test_ans = np.array(
        [[0., 0., 0., 1., 0., 0., 0., 0., 1., 0., ],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., ],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., ],
         [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., ],
         [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., ],
         [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
         [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., ],
         [1., 0., 0., 0., 1., 1., 0., 0., 0., 1., ],
         [0., 1., 1., 0., 1., 0., 0., 0., 1., 0., ]]
    )
    word2Ind_ans = {'All': 0, "All's": 1, 'END': 2, 'START': 3, 'ends': 4, 'glitters': 5, 'gold': 6, "isn't": 7,
                    'that': 8, 'well': 9}

    # Test correct word2Ind
    assert (word2Ind_ans == word2Ind_test), "Your word2Ind is incorrect:\nCorrect: {}\nYours: {}".format(word2Ind_ans,
                                                                                                         word2Ind_test)

    # Test correct M shape
    assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(
        M_test.shape, M_test_ans.shape)

    # Test correct M values
    for w1 in word2Ind_ans.keys():
        idx1 = word2Ind_ans[w1]
        for w2 in word2Ind_ans.keys():
            idx2 = word2Ind_ans[w2]
            student = M_test[idx1, idx2]
            correct = M_test_ans[idx1, idx2]
            if student != correct:
                print("Correct M:")
                print(M_test_ans)
                print("Your M: ")
                print(M_test)
                raise AssertionError(
                    "Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(
                        idx1, idx2, w1, w2, student, correct))

    # Print Success
    print("-" * 80)
    print("Passed All Tests!")
    print("-" * 80)

def test_reduced_SVD():
    test_corpus = ["START All that glitters isn't gold END".split(" "),
                   "START All's well that ends well END".split(" ")]
    M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
    M_test_reduced = reduce_to_k_dim(M_test, k=2)

    # Test proper dimensions
    assert (M_test_reduced.shape[0] == 10), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 10)
    assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

    # Print Success
    print("-" * 80)
    print("Passed All Tests!")
    print("-" * 80)




test_reduced_SVD()