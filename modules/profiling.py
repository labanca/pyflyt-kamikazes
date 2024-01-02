import cProfile

def main():
    import apps.run_parallel


if __name__ == "__main__":
    # Create a cProfile object
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    # Run your main code
    main()

    # Stop profiling
    profiler.disable()

    # Print the profiling results
    profiler.print_stats()