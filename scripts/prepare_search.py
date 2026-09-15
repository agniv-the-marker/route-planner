"""Build the derived, hash-keyed CPU street search index without starting a GPU."""
from src.street_search import StreetSearch

if __name__ == '__main__':
    search = StreetSearch.load_prepared()
    print(f'Prepared {len(search.xy):,} locations and {len(search.edges):,} directed pieces in {search.index_seconds:.2f}s')
