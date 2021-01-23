from data_utils import DataUtils

dataUtils = DataUtils()

if __name__ == '__main__':

    dataUtils.pre_process_aws(0.01)
    dataUtils.pre_process_aws(0.1)
    dataUtils.pre_process_aws(1)

    dataUtils.pre_process_col_tweets()
