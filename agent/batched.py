from typing import ClassVar, Generator, Iterable, TypeVar


T = TypeVar("T")


class Batched:
    DEFAULT_BATCH_SIZE: ClassVar[int] = 512

    @staticmethod
    def iter(
        elements: Iterable[T],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Generator[list[T], None, None]:
        batch = []
        for element in elements:
            batch.append(element)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
