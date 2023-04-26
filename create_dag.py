n_jobs = 311  # 2848
data_file = "/path/test_dataset_5"
log_path = "/crea_cartella"

with open("jobs.dag", "w") as f:
    for i in range(n_jobs):
        f.write("JOB {:d} single_job.sub\n"
                "RETRY {:d} 1\n"
                "VARS {:d} data_file=\"{}\" id=\"{:d}\" log_path=\"{}\"\n\n".format(i + 1,
                                                                                    i + 1,
                                                                                    i + 1,
                                                                                    data_file,
                                                                                    i,
                                                                                    log_path))
