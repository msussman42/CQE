def generate_rank_indexes(num_jobs, num_ranks, rank, comm):

    from numpy import arange

    jobs_per_rank = num_jobs // num_ranks
    leftover = num_jobs % num_ranks
    if rank < leftover:
        jobs_per_rank += 1
    jobsizes = comm.allgather(jobs_per_rank)
    starts = list(sum(jobsizes[:i]) for i in range(len(jobsizes)))
    idxs = arange(starts[rank], starts[rank] + jobsizes[rank])

    return idxs
