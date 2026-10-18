#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
    - prvo prognoza;
    - zatim ocena nakon poznatog ishoda;
    - zatim ažuriranje pondera i eksperata;
    - vremenske skale 30 i 120 kola;
    - korekcija zajedničkog skora odnosima parova;
    - deterministička pretraga kombinacija.

Eksperti:
    1. Regresija distribucije sa celom istorijom.
    2. Regresija distribucije sa poluživotom 30 kola.
    3. Regresija distribucije sa poluživotom 120 kola.

Treći ekspert regresira i 741 odnos parova.

Izlaz regresije:
    Kumulativne distribucije sedam sortiranih pozicija.
    Trenira se kvadratnim regresionim gubitkom.

Kontekst:
    Opis poslednjeg izvlačenja i prosečni distribucioni opisi
    prethodna 4, 16 i 32 izvlačenja.

Sva ranija kola ulaze u statistike učenja.
Prozor konteksta nije ograničenje dužine trening istorije.
"""


import os

# Ograničenje važi samo za ovaj proces.
for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Potrebna biblioteka: numpy.\n"
        "Instalacija: python3 -m pip install numpy"
    ) from exc


DEFAULT_FILES = (
    "/data/loto7_4704_k83.csv",
    "/data/loto7_4704_k83_loto_2974.csv",
    "/data/loto7_4704_k83_loto_plus_1730.csv",
)

CONTEXT = 32
WARMUP = 120
VALIDATION = 48
TEST = 64
MIN_ROWS = CONTEXT + WARMUP + VALIDATION + TEST

RIDGE = 0.03
ETA = 0.25
PAIR_WEIGHT = 0.20
BEAM_WIDTH = 256

GRID = np.arange(1, 40)
PAIR_I, PAIR_J = np.triu_indices(39, 1)
DRAW_PAIR_I, DRAW_PAIR_J = np.triu_indices(7, 1)

CDF_SIZE = 7 * 39
PAIR_SIZE = len(PAIR_I)

EXPERT_NAMES = (
    "Cela_istorija",
    "Poluzivot_30",
    "Poluzivot_120",
)


def load_csv(filename):
    path = Path(filename).expanduser().resolve(strict=True)
    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or all(not value.strip() for value in row):
                continue

            if len(row) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    f"očekivano 7 kolona; dobijeno {len(row)}."
                )

            try:
                numbers = [int(value.strip()) for value in row]
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "sve vrednosti moraju biti celi brojevi."
                ) from exc

            if len(set(numbers)) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: ponovljen broj."
                )

            if min(numbers) < 1 or max(numbers) > 39:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "broj izvan opsega 1..39."
                )

            rows.append(sorted(numbers))

    if len(rows) < MIN_ROWS:
        raise ValueError(
            f"{path.name}: potrebno najmanje {MIN_ROWS} izvlačenja."
        )

    draws = np.asarray(rows, dtype=np.int64)

    canonical = "\n".join(
        ",".join(map(str, draw)) for draw in draws
    ).encode("ascii")

    fingerprint = hashlib.sha256(canonical).hexdigest()
    return path, draws, fingerprint


def describe_draw(draw):
    numbers = draw.astype(float)

    distances = np.abs(
        numbers[:, None] - numbers[None, :]
    )[DRAW_PAIR_I, DRAW_PAIR_J]

    return np.r_[
        numbers / 39.0,
        [np.mean(distances <= limit) for limit in (2, 5, 10, 20)],
        (numbers[-1] - numbers[0]) / 38.0,
    ]


def make_data(draws):
    profiles = np.asarray(
        [describe_draw(draw) for draw in draws],
        dtype=float,
    )

    features = []

    for t in range(CONTEXT, len(draws) + 1):
        features.append(np.r_[
            1.0,
            profiles[t - 1],
            profiles[t - 4:t].mean(axis=0),
            profiles[t - 16:t].mean(axis=0),
            profiles[t - 32:t].mean(axis=0),
        ])

    x = np.asarray(features, dtype=float)
    targets = draws[CONTEXT:]

    cdf_targets = (
        targets[:, :, None] <= GRID[None, None, :]
    ).astype(float).reshape(len(targets), CDF_SIZE)

    membership = np.zeros((len(targets), 39), dtype=float)
    membership[
        np.arange(len(targets))[:, None], targets - 1
    ] = 1.0

    pair_targets = (
        membership[:, PAIR_I] * membership[:, PAIR_J]
    )

    return x, targets, cdf_targets, pair_targets


class DistributionExpert:
    """
    Regresija iz eksponencijalno ponderisanih dovoljnih statistika.

    Nema odsecanja istorijskih redova.
    Ridge regularizacija ostaje prisutna pri svakoj prognozi.
    """

    def __init__(self, dimension, outputs, half_life=None):
        self.half_life = half_life

        self.decay = (
            1.0 if half_life is None
            else math.exp(math.log(0.5) / half_life)
        )

        self.gram = np.zeros((dimension, dimension), dtype=float)
        self.cross = np.zeros((dimension, outputs), dtype=float)

        self.mass = 0.0
        self.observations = 0

        self.penalty = np.full(dimension, RIDGE, dtype=float)
        self.penalty[0] = 1e-9

    def predict(self, x):
        if self.observations == 0:
            raise ValueError("Ekspert još nema trening opažanja.")

        matrix = self.gram / self.mass
        matrix = matrix.copy()
        matrix.flat[::matrix.shape[0] + 1] += self.penalty

        # Za jednu prognozu potreban je jedan vektorski sistem.
        projection = np.linalg.solve(matrix, x)
        result = projection @ (self.cross / self.mass)

        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Nevažeći regresioni izlaz.")

        return result

    def update(self, x, target):
        self.gram *= self.decay
        self.gram += np.outer(x, x)

        self.cross *= self.decay
        self.cross += np.outer(x, target)

        self.mass = self.decay * self.mass + 1.0
        self.observations += 1


def isotonic(values):
    """PAVA: projekcija na neopadajući niz."""
    means = []
    sizes = []

    for value in np.clip(values, 0.0, 1.0):
        means.append(float(value))
        sizes.append(1)

        while len(means) >= 2 and means[-2] > means[-1]:
            total = sizes[-2] + sizes[-1]

            combined = (
                means[-2] * sizes[-2]
                + means[-1] * sizes[-1]
            ) / total

            means[-2:] = [combined]
            sizes[-2:] = [total]

    return np.repeat(means, sizes)


def distribution_from_regression(prediction):
    """
    Pretvara regresioni izlaz u validne diskretne distribucije.

    Dozvoljeni opseg sortirane pozicije proizlazi isključivo
    iz pravila da kombinacija ima sedam različitih brojeva.
    """
    raw = np.asarray(prediction, dtype=float).reshape(7, 39)

    if not np.all(np.isfinite(raw)):
        raise FloatingPointError("Nevažeći CDF izlaz.")

    pmf = np.zeros((7, 39), dtype=float)

    for position in range(7):
        corrected = isotonic(raw[position])
        corrected[-1] = 1.0

        mass = np.maximum(
            np.diff(np.r_[0.0, corrected]), 0.0
        )

        minimum = position + 1
        maximum = 33 + position

        mass[(GRID < minimum) | (GRID > maximum)] = 0.0
        total = mass.sum()

        if total <= 0 or not math.isfinite(total):
            raise FloatingPointError(
                "Regresija nije dala pozitivnu masu u legalnom "
                f"opsegu sortirane pozicije {position + 1}."
            )

        pmf[position] = mass / total

    return pmf


def expert_weights(log_weights):
    shifted = log_weights - np.max(log_weights)
    values = np.exp(shifted)
    return values / values.sum()


def pair_modifier(pair_prediction, mixed_pmf):
    """
    Bounded korekcija, analogna korekciji parova iz projekta.

    Referentni skor izvodi se iz aktuelne distribucije ansambla,
    bez uniformnog eksperta.

    Ovo su modeli odnosa parova za rangiranje kombinacija,
    a ne kalibrisane verovatnoće dobitka.
    """
    inclusion_score = np.minimum(mixed_pmf.sum(axis=0), 1.0)

    reference = (
        inclusion_score[PAIR_I]
        * inclusion_score[PAIR_J]
        * (6.0 / 7.0)
    )

    estimated = np.clip(pair_prediction, 0.0, 1.0)

    ratio = np.log(
        (estimated + 1e-12) / (reference + 1e-12)
    )
    ratio = np.clip(ratio, -1.0, 1.0)

    matrix = np.zeros((39, 39), dtype=float)
    matrix[PAIR_I, PAIR_J] = ratio
    matrix[PAIR_J, PAIR_I] = ratio

    return matrix


def select_combination(pmf, modifier):
    """
    Beam pretraga modelskog skora.
    Ne zaokružuje regresione izlaze i ne prepravlja rezultat.
    """
    log_pmf = np.full((7, 39), -np.inf, dtype=float)
    np.log(pmf, out=log_pmf, where=pmf > 0)

    beam = [((), 0.0)]

    for position in range(7):
        candidates = []

        for prefix, previous_score in beam:
            first = prefix[-1] + 1 if prefix else 1

            for number in range(first, 34 + position):
                log_mass = log_pmf[position, number - 1]

                if not math.isfinite(float(log_mass)):
                    continue

                relationship = sum(
                    modifier[number - 1, before - 1]
                    for before in prefix
                )

                score = (
                    previous_score
                    + float(log_mass)
                    + PAIR_WEIGHT * float(relationship)
                )

                candidates.append((
                    prefix + (number,),
                    score,
                ))

        if not candidates:
            raise FloatingPointError(
                "Distribucija i trenutni beam nisu dali legalnu "
                "kompletnu kombinaciju. Nema zamenskog generisanja."
            )

        candidates.sort(key=lambda item: (-item[1], item[0]))
        beam = candidates[:BEAM_WIDTH]

    return beam[0]


def longest_run(combination):
    maximum = current = 1

    for previous, number in zip(combination[:-1], combination[1:]):
        current = current + 1 if number == previous + 1 else 1
        maximum = max(maximum, current)

    return maximum


def report_check(predictions, hits, control, losses, pair_losses):
    hits = np.asarray(hits)
    control = np.asarray(control)

    print("\nZAVRŠNA ONLINE PROVERA — POSLEDNJA 64 KOLA")
    print(f"Prosečno pogodaka: {hits.mean():.4f}")
    print(
        "Raspodela pogodaka 0..7: "
        f"{np.bincount(hits, minlength=8).tolist()}"
    )
    print(f"Kola sa 3+: {int(np.sum(hits >= 3))}")
    print(f"Kola sa 4+: {int(np.sum(hits >= 4))}")

    print(
        "Kontrola: prethodno izvuceno kolo — "
        f"{control.mean():.4f}"
    )
    print(
        "Razlika prema kontroli: "
        f"{hits.mean() - control.mean():+.4f}"
    )

    print(f"Prosečni kvadratni CDF gubitak: {np.mean(losses):.7f}")
    print(f"Prosečni kvadratni gubitak parova: {np.mean(pair_losses):.7f}")
    print(f"Različite prognoze: {len(set(predictions))}/{len(predictions)}")

    print(
        "Prognoze sa 5+ uzastopnih brojeva: "
        f"{sum(longest_run(item) >= 5 for item in predictions)}"
    )
    print(
        "Prognoze potpuno u jednoj polovini: "
        f"{sum(max(item) <= 19 or min(item) >= 20 for item in predictions)}"
    )


def process_dataset(path, draws, fingerprint):
    x, targets, cdf_targets, pair_targets = make_data(draws)

    count = len(targets)
    test_start = count - TEST
    validation_start = test_start - VALIDATION

    dimension = x.shape[1]

    experts = [
        DistributionExpert(dimension, CDF_SIZE, half_life=None),
        DistributionExpert(dimension, CDF_SIZE, half_life=30),
        DistributionExpert(
            dimension,
            CDF_SIZE + PAIR_SIZE,
            half_life=120,
        ),
    ]

    log_weights = np.zeros(3, dtype=float)

    predictions = []
    test_hits = []
    control_hits = []
    test_losses = []
    pair_losses = []
    validation_losses = []

    total_expert_losses = np.zeros(3)
    scored_rounds = 0
    copied_previous = 0

    print("\n" + "=" * 76)
    print(path.name)
    print(f"Izvlačenja: {len(draws)}")
    print(f"Istorijskih prelaza: {count}")
    print(f"SHA-256: {fingerprint}")
    print(
        "Kontrolni blok od 48 kola — CSV redovi: "
        f"{CONTEXT + validation_start + 1}..{CONTEXT + test_start}"
    )
    print(
        "Završna provera — CSV redovi: "
        f"{CONTEXT + test_start + 1}..{len(draws)}"
    )
    print("Parametri su fiksni; kontrolni blok ih ne podešava.")

    final_pmf = None
    final_pairs = None
    final_weights = None

    for index in range(count + 1):
        if index >= WARMUP:
            # 1. PROGNOZA: samo prethodno ažurirana stanja.
            outputs = [
                expert.predict(x[index])
                for expert in experts
            ]

            component_pmfs = np.asarray([
                distribution_from_regression(output[:CDF_SIZE])
                for output in outputs
            ])

            component_cdfs = component_pmfs.cumsum(axis=-1)
            weights = expert_weights(log_weights)

            mixed_pmf = np.tensordot(
                weights, component_pmfs, axes=(0, 0)
            )

            mixed_cdf = mixed_pmf.cumsum(axis=-1)
            pair_prediction = outputs[2][CDF_SIZE:]

            if index == count:
                final_pmf = mixed_pmf
                final_pairs = pair_prediction
                final_weights = weights
                break

            target_cdf = cdf_targets[index].reshape(7, 39)

            # 2. OCENA prognoze, pre učenja tog izvlačenja.
            losses = np.mean(
                (component_cdfs - target_cdf[None, :, :]) ** 2,
                axis=(1, 2),
            )

            mixed_loss = float(np.mean(
                (mixed_cdf - target_cdf) ** 2
            ))

            total_expert_losses += losses
            scored_rounds += 1

            if validation_start <= index < test_start:
                validation_losses.append(mixed_loss)

            if index >= test_start:
                modifier = pair_modifier(
                    pair_prediction, mixed_pmf
                )

                combination, _ = select_combination(
                    mixed_pmf, modifier
                )

                actual = targets[index]
                previous = draws[CONTEXT + index - 1]

                predictions.append(combination)
                test_hits.append(
                    len(set(combination).intersection(actual))
                )
                control_hits.append(
                    len(set(previous).intersection(actual))
                )

                copied_previous += (
                    tuple(combination) == tuple(previous)
                )

                test_losses.append(mixed_loss)
                pair_losses.append(float(np.mean(
                    (pair_prediction - pair_targets[index]) ** 2
                )))

                if len(predictions) % 8 == 0:
                    print(
                        f"  Provera {len(predictions)}/{TEST}",
                        flush=True,
                    )

            # 3. AŽURIRANJE PONDERA tek nakon ocene.
            log_weights -= ETA * losses
            log_weights -= np.max(log_weights)

        if index == count:
            break

        # 4. AŽURIRANJE EKSPERATA stvarnim ishodom.
        experts[0].update(x[index], cdf_targets[index])
        experts[1].update(x[index], cdf_targets[index])

        experts[2].update(
            x[index],
            np.r_[cdf_targets[index], pair_targets[index]],
        )

        if (index + 1) % 500 == 0:
            print(
                f"  Obrađeno istorijskih prelaza: {index + 1}/{count}",
                flush=True,
            )

    if final_pmf is None:
        raise ValueError("Nije formirana završna distribucija.")

    report_check(
        predictions,
        test_hits,
        control_hits,
        test_losses,
        pair_losses,
    )

    print(f"Kopije prethodnog kola: {copied_previous}")
    print(
        "CDF gubitak prethodnog kontrolnog bloka od 48 kola: "
        f"{np.mean(validation_losses):.7f}"
    )

    print("\nEKSPERTI — SVI PRETHODNO OCENJENI PRELAZI")

    for name, expert, loss, weight in zip(
        EXPERT_NAMES,
        experts,
        total_expert_losses / scored_rounds,
        final_weights,
    ):
        print(
            f"  {name}: "
            f"ponder={weight:.6f}, "
            f"CDF gubitak={loss:.7f}, "
            f"naučenih prelaza={expert.observations}, "
            f"efektivna masa={expert.mass:.2f}"
        )

    final_modifier = pair_modifier(final_pairs, final_pmf)
    combination, score = select_combination(
        final_pmf, final_modifier
    )

    print("\nFINALNI NEXT: " + ", ".join(map(str, combination)))
    print(f"Modelski skor za rangiranje: {score:.8f}")
    print(f"Najduži uzastopni niz: {longest_run(combination)}")

    lower = sum(number <= 19 for number in combination)
    print(f"Brojeva 1..19 / 20..39: {lower} / {7 - lower}")
    print(f"Širina determinističke beam pretrage: {BEAM_WIDTH}")
    print("Beam pretraga nije iscrpna pretraga svih kombinacija.")

    return combination


def main():
    parser = argparse.ArgumentParser(
        description="Online ansambl eksperata za tri Loto 7/39 CSV-a."
    )
    parser.add_argument("csv", nargs="*")
    args = parser.parse_args()

    filenames = args.csv or DEFAULT_FILES

    if len(filenames) != 3:
        raise ValueError("Navedi tačno tri CSV putanje.")

    datasets = [load_csv(filename) for filename in filenames]

    if len({path for path, _, _ in datasets}) != 3:
        raise ValueError("Potrebna su tri različita CSV fajla.")

    print("ONLINE EXPERTS LOTO V1")
    print("Cela istorija + poluživot 30 + poluživot 120.")
    print("Prognoza → ocena → ponderi → učenje ishoda.")
    print("Tri nezavisna CSV-a; jedna završna kombinacija po CSV-u.")
    print("Bez random generisanja i uniformnog eksperta.")
    print("CSV redosled mora biti: najstarije → najnovije.")

    results = []

    for path, draws, fingerprint in datasets:
        combination = process_dataset(path, draws, fingerprint)
        results.append((path.name, combination))

    print("\n" + "=" * 76)
    print("ZAVRŠNI NEXT PREGLED")

    for name, combination in results:
        print(f"{name}: " + ", ".join(map(str, combination)))

    grouped = {}

    for name, combination in results:
        grouped.setdefault(combination, []).append(name)

    for combination, names in grouped.items():
        if len(names) > 1:
            print(
                "\nIsti NEXT: "
                + " / ".join(names)
                + ": "
                + ", ".join(map(str, combination))
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrekinuto.", file=sys.stderr)
        sys.exit(130)
    except (
        OSError,
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
        csv.Error,
    ) as exc:
        print(f"\nGREŠKA: {exc}", file=sys.stderr)
        sys.exit(1)



"""
ONLINE EXPERTS LOTO V1
Cela istorija + poluživot 30 + poluživot 120.
Prognoza → ocena → ponderi → učenje ishoda.
Tri nezavisna CSV-a; jedna završna kombinacija po CSV-u.
Bez random generisanja i uniformnog eksperta.
CSV redosled mora biti: najstarije → najnovije.


============================================================================
loto7_4704_k83.csv
Izvlačenja: 4704
Istorijskih prelaza: 4672
SHA-256: 1decd748cc0c398e384d9b16fd7e7eb50a5396f59a2f2cf9e7cdd05bdd07ae09
Kontrolni blok od 48 kola — CSV redovi: 4593..4640
Završna provera — CSV redovi: 4641..4704
Parametri su fiksni; kontrolni blok ih ne podešava.
  Obrađeno istorijskih prelaza: 500/4672
  Obrađeno istorijskih prelaza: 1000/4672
  Obrađeno istorijskih prelaza: 1500/4672
  Obrađeno istorijskih prelaza: 2000/4672
  Obrađeno istorijskih prelaza: 2500/4672
  Obrađeno istorijskih prelaza: 3000/4672
  Obrađeno istorijskih prelaza: 3500/4672
  Obrađeno istorijskih prelaza: 4000/4672
  Obrađeno istorijskih prelaza: 4500/4672
  Provera 8/64
  Provera 16/64
  Provera 24/64
  Provera 32/64
  Provera 40/64
  Provera 48/64
  Provera 56/64
  Provera 64/64

ZAVRŠNA ONLINE PROVERA — POSLEDNJA 64 KOLA
Prosečno pogodaka: 1.1250
Raspodela pogodaka 0..7: [21, 19, 19, 5, 0, 0, 0, 0]
Kola sa 3+: 5
Kola sa 4+: 0
Kontrola: prethodno izvuceno kolo — 1.1406
Razlika prema kontroli: -0.0156
Prosečni kvadratni CDF gubitak: 0.0732596
Prosečni kvadratni gubitak parova: 0.0277670
Različite prognoze: 52/64
Prognoze sa 5+ uzastopnih brojeva: 0
Prognoze potpuno u jednoj polovini: 0
Kopije prethodnog kola: 0
CDF gubitak prethodnog kontrolnog bloka od 48 kola: 0.0696929

EKSPERTI — SVI PRETHODNO OCENJENI PRELAZI
  Cela_istorija: ponder=0.566788, CDF gubitak=0.0732976, naučenih prelaza=4672, efektivna masa=4672.00
  Poluzivot_30: ponder=0.065887, CDF gubitak=0.0751887, naučenih prelaza=4672, efektivna masa=43.78
  Poluzivot_120: ponder=0.367326, CDF gubitak=0.0736787, naučenih prelaza=4672, efektivna masa=173.62

FINALNI NEXT: 1, x, 14, y, 29, z, 39
Modelski skor za rangiranje: -15.19666851
Najduži uzastopni niz: 1
Brojeva 1..19 / 20..39: 4 / 3
Širina determinističke beam pretrage: 256
Beam pretraga nije iscrpna pretraga svih kombinacija.



============================================================================
loto7_4704_k83_loto_2974.csv
Izvlačenja: 2974
Istorijskih prelaza: 2942
SHA-256: fc5313380978e4826789413029140d99f09bd92c143892bf9b9027ff9f276be7
Kontrolni blok od 48 kola — CSV redovi: 2863..2910
Završna provera — CSV redovi: 2911..2974
Parametri su fiksni; kontrolni blok ih ne podešava.
  Obrađeno istorijskih prelaza: 500/2942
  Obrađeno istorijskih prelaza: 1000/2942
  Obrađeno istorijskih prelaza: 1500/2942
  Obrađeno istorijskih prelaza: 2000/2942
  Obrađeno istorijskih prelaza: 2500/2942
  Provera 8/64
  Provera 16/64
  Provera 24/64
  Provera 32/64
  Provera 40/64
  Provera 48/64
  Provera 56/64
  Provera 64/64

ZAVRŠNA ONLINE PROVERA — POSLEDNJA 64 KOLA
Prosečno pogodaka: 1.2188
Raspodela pogodaka 0..7: [14, 27, 18, 5, 0, 0, 0, 0]
Kola sa 3+: 5
Kola sa 4+: 0
Kontrola: prethodno izvuceno kolo — 1.3125
Razlika prema kontroli: -0.0938
Prosečni kvadratni CDF gubitak: 0.0729731
Prosečni kvadratni gubitak parova: 0.0277447
Različite prognoze: 57/64
Prognoze sa 5+ uzastopnih brojeva: 0
Prognoze potpuno u jednoj polovini: 0
Kopije prethodnog kola: 1
CDF gubitak prethodnog kontrolnog bloka od 48 kola: 0.0699419

EKSPERTI — SVI PRETHODNO OCENJENI PRELAZI
  Cela_istorija: ponder=0.505287, CDF gubitak=0.0733188, naučenih prelaza=2942, efektivna masa=2942.00
  Poluzivot_30: ponder=0.119607, CDF gubitak=0.0753612, naučenih prelaza=2942, efektivna masa=43.78
  Poluzivot_120: ponder=0.375106, CDF gubitak=0.0737411, naučenih prelaza=2942, efektivna masa=173.62

FINALNI NEXT: 1, x, 10, y, 25, z, 39
Modelski skor za rangiranje: -15.46794726
Najduži uzastopni niz: 1
Brojeva 1..19 / 20..39: 4 / 3
Širina determinističke beam pretrage: 256
Beam pretraga nije iscrpna pretraga svih kombinacija.



============================================================================
loto7_4704_k83_loto_plus_1730.csv
Izvlačenja: 1730
Istorijskih prelaza: 1698
SHA-256: 1f4252375372cbbb88cd026892dc19eb5400e6386bfc5dc613ff7d19d9db0504
Kontrolni blok od 48 kola — CSV redovi: 1619..1666
Završna provera — CSV redovi: 1667..1730
Parametri su fiksni; kontrolni blok ih ne podešava.
  Obrađeno istorijskih prelaza: 500/1698
  Obrađeno istorijskih prelaza: 1000/1698
  Obrađeno istorijskih prelaza: 1500/1698
  Provera 8/64
  Provera 16/64
  Provera 24/64
  Provera 32/64
  Provera 40/64
  Provera 48/64
  Provera 56/64
  Provera 64/64

ZAVRŠNA ONLINE PROVERA — POSLEDNJA 64 KOLA
Prosečno pogodaka: 1.2031
Raspodela pogodaka 0..7: [16, 26, 17, 3, 2, 0, 0, 0]
Kola sa 3+: 5
Kola sa 4+: 2
Kontrola: prethodno izvuceno kolo — 1.2812
Razlika prema kontroli: -0.0781
Prosečni kvadratni CDF gubitak: 0.0728270
Prosečni kvadratni gubitak parova: 0.0277673
Različite prognoze: 37/64
Prognoze sa 5+ uzastopnih brojeva: 0
Prognoze potpuno u jednoj polovini: 0
Kopije prethodnog kola: 3
CDF gubitak prethodnog kontrolnog bloka od 48 kola: 0.0760558

EKSPERTI — SVI PRETHODNO OCENJENI PRELAZI
  Cela_istorija: ponder=0.415834, CDF gubitak=0.0732498, naučenih prelaza=1698, efektivna masa=1698.00
  Poluzivot_30: ponder=0.211980, CDF gubitak=0.0749578, naučenih prelaza=1698, efektivna masa=43.78
  Poluzivot_120: ponder=0.372186, CDF gubitak=0.0735309, naučenih prelaza=1698, efektivna masa=173.61

FINALNI NEXT: 1, x, 11, y, 28, z, 39
Modelski skor za rangiranje: -15.40510408
Najduži uzastopni niz: 1
Brojeva 1..19 / 20..39: 4 / 3
Širina determinističke beam pretrage: 256
Beam pretraga nije iscrpna pretraga svih kombinacija.


============================================================================
ZAVRŠNI NEXT PREGLED
loto7_4704_k83.csv: 1, x, 14, y, 29, z, 39
loto7_4704_k83_loto_2974.csv: 1, x, 10, y, 25, z, 39
loto7_4704_k83_loto_plus_1730.csv: 1, x, 11, y, 28, z, 39
"""



"""
adaptivno ponderisanje eksperata

Prvo napraviti prognozu, zatim je oceniti nakon izvlačenja i tek onda ažurira modele i njihove pondere. 
Eksperti koji su prethodno imali manju grešku dobijaju veću težinu. To je ključna ideja.
Prilagoditi eksperte distribucijama kombinacija, objediniti izlaz na 7/39 i obraditi svaki ceo CSV nezavisno. 

Online učenje pondera, različite vremenske skale, korekcija odnosa parova i deterministička pretraga. 
Svako kolo prvo se prognozira i oceni, pa tek onda ulazi u ažuriranje. 
Završni NEXT koristi sve istorijske prelaze odgovarajućeg CSV-a.
"""
