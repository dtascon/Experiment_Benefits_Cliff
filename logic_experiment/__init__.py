from otree.api import *
import csv
import random
import string
from pathlib import Path

doc = """
Enhanced PRD / Benefit Cliff Experiment - 60 Strategic Vignettes
"""

# ----------------------------
# Configuration
# ----------------------------

SEED = 123
N_VIGNETTES = 60

WEEKS_PER_MONTH = 52 / 12
PAYROLL_TAX_RATE = 0.0765
INCOME_TAX_RATE = 0.10


# ----------------------------
# Location and Family-Specific Benefit Rules
# ----------------------------

class LocationBenefits:
    """Location and family-specific benefit rules based on PRD Dashboard"""

    def __init__(self, state="PA", county="Allegheny", num_children=2, child_ages=None, has_disability=False):
        self.state = state
        self.county = county
        self.num_children = num_children
        self.child_ages = child_ages or [3, 6]
        self.has_disability = has_disability
        self.set_rules()

    def set_rules(self):
        """Set state/county specific parameters"""

        if self.state == "PA":
            self.snap_max_gross = 2830.0 if self.num_children > 0 else 1580.0
            self.snap_net_limit = 2177.0 if self.num_children > 0 else 1215.0
            self.snap_max_benefit = 740.0 if self.num_children > 0 else 291.0
            self.medicaid_value = 450.0
            self.medicaid_adult_limit = 1806.0
            self.medicaid_child_limit = 4168.0
            self.ccdf_max = 800.0
            self.ccdf_copay_threshold = 1800.0
            self.ccdf_exit = 3200.0
            self.tanf_max = 403.0
            self.tanf_limit = 614.0

        elif self.state == "GA":
            self.snap_max_gross = 2830.0 if self.num_children > 0 else 1580.0
            self.snap_net_limit = 2177.0 if self.num_children > 0 else 1215.0
            self.snap_max_benefit = 740.0 if self.num_children > 0 else 291.0
            self.medicaid_value = 450.0
            self.medicaid_adult_limit = 503.0
            self.medicaid_child_limit = 4168.0
            self.ccdf_max = 750.0
            self.ccdf_copay_threshold = 1600.0
            self.ccdf_exit = 3000.0
            self.tanf_max = 280.0
            self.tanf_limit = 500.0

        elif self.state == "TX":
            self.snap_max_gross = 2830.0 if self.num_children > 0 else 1580.0
            self.snap_net_limit = 2177.0 if self.num_children > 0 else 1215.0
            self.snap_max_benefit = 740.0 if self.num_children > 0 else 291.0
            self.medicaid_value = 450.0
            self.medicaid_adult_limit = 288.0
            self.medicaid_child_limit = 3629.0
            self.ccdf_max = 700.0
            self.ccdf_copay_threshold = 1500.0
            self.ccdf_exit = 2800.0
            self.tanf_max = 303.0
            self.tanf_limit = 450.0

        elif self.state == "NY":
            self.snap_max_gross = 2830.0 if self.num_children > 0 else 1580.0
            self.snap_net_limit = 2177.0 if self.num_children > 0 else 1215.0
            self.snap_max_benefit = 740.0 if self.num_children > 0 else 291.0
            self.medicaid_value = 450.0
            self.medicaid_adult_limit = 1806.0
            self.medicaid_child_limit = 4168.0
            self.ccdf_max = 900.0
            self.ccdf_copay_threshold = 2000.0
            self.ccdf_exit = 3500.0
            self.tanf_max = 789.0
            self.tanf_limit = 800.0

        elif self.state == "OH":
            self.snap_max_gross = 2830.0 if self.num_children > 0 else 1580.0
            self.snap_net_limit = 2177.0 if self.num_children > 0 else 1215.0
            self.snap_max_benefit = 740.0 if self.num_children > 0 else 291.0
            self.medicaid_value = 450.0
            self.medicaid_adult_limit = 1806.0
            self.medicaid_child_limit = 4168.0
            self.ccdf_max = 750.0
            self.ccdf_copay_threshold = 1700.0
            self.ccdf_exit = 3100.0
            self.tanf_max = 385.0
            self.tanf_limit = 600.0

        self.eitc_phase_in = 0.40 if self.num_children >= 3 else (0.34 if self.num_children == 2 else 0.15)
        self.eitc_max = 6935.0 if self.num_children >= 3 else (5920.0 if self.num_children == 2 else 3584.0)
        self.eitc_plateau_start = 15950.0 / 12
        self.eitc_phase_out_start = 25500.0 / 12
        self.eitc_phase_out_rate = 0.2106
        self.eitc_phase_out_end = 57000.0 / 12

        if self.has_disability:
            self.ssdi_benefit = 1537.0
            self.ssi_benefit = 943.0

    def snap(self, monthly_gross: float) -> float:
        if monthly_gross > self.snap_max_gross:
            return 0.0
        net_income = monthly_gross - 200
        if net_income > self.snap_net_limit:
            return 0.0
        benefit = self.snap_max_benefit - (0.30 * max(0, net_income))
        return max(0.0, round(benefit, 2))

    def medicaid(self, monthly_gross: float) -> float:
        if self.num_children > 0:
            if monthly_gross <= self.medicaid_adult_limit:
                return self.medicaid_value * (1 + self.num_children * 0.3)
            elif monthly_gross <= self.medicaid_child_limit:
                return self.medicaid_value * (self.num_children * 0.3)
        else:
            if monthly_gross <= self.medicaid_adult_limit:
                return self.medicaid_value
        return 0.0

    def ccdf(self, monthly_gross: float) -> float:
        if self.num_children == 0 or all(age >= 13 for age in self.child_ages):
            return 0.0
        if monthly_gross <= self.ccdf_copay_threshold:
            return self.ccdf_max
        elif monthly_gross <= self.ccdf_exit:
            phase_range = self.ccdf_exit - self.ccdf_copay_threshold
            phase_amount = ((monthly_gross - self.ccdf_copay_threshold) / phase_range) * self.ccdf_max
            return max(0.0, round(self.ccdf_max - phase_amount, 2))
        return 0.0

    def tanf(self, monthly_gross: float) -> float:
        if self.num_children == 0:
            return 0.0
        if monthly_gross > self.tanf_limit:
            return 0.0
        benefit = self.tanf_max - (0.50 * monthly_gross)
        return max(0.0, round(benefit, 2))

    def wic(self, monthly_gross: float) -> float:
        if self.num_children == 0 or all(age >= 5 for age in self.child_ages):
            return 0.0
        wic_limit = 2177.0
        if monthly_gross <= wic_limit:
            return 50.0
        return 0.0

    def head_start(self, monthly_gross: float) -> float:
        has_preschooler = any(3 <= age <= 5 for age in self.child_ages)
        if not has_preschooler:
            return 0.0
        head_start_limit = 2177.0
        if monthly_gross <= head_start_limit:
            return 300.0
        return 0.0

    def eitc(self, annual_gross: float) -> float:
        if self.num_children == 0:
            return 0.0
        monthly = annual_gross / 12
        if monthly < self.eitc_plateau_start:
            credit = annual_gross * self.eitc_phase_in
        elif monthly < self.eitc_phase_out_start:
            credit = self.eitc_max
        elif monthly < self.eitc_phase_out_end:
            excess = annual_gross - (self.eitc_phase_out_start * 12)
            credit = self.eitc_max - (excess * self.eitc_phase_out_rate)
        else:
            credit = 0.0
        return max(0.0, round(credit / 12, 2))

    def disability_benefits(self, monthly_gross: float) -> dict:
        if not self.has_disability:
            return {"ssdi": 0.0, "ssi": 0.0}
        ssdi = self.ssdi_benefit
        ssi_limit = 943.0
        if monthly_gross < 1000:
            ssi = max(0.0, ssi_limit - (0.50 * monthly_gross))
        else:
            ssi = 0.0
        return {"ssdi": ssdi, "ssi": ssi}

    def total_benefits(self, monthly_gross: float) -> dict:
        annual_gross = monthly_gross * 12
        snap = self.snap(monthly_gross)
        med = self.medicaid(monthly_gross)
        ccdf = self.ccdf(monthly_gross)
        tanf = self.tanf(monthly_gross)
        wic = self.wic(monthly_gross)
        head_start = self.head_start(monthly_gross)
        eitc = self.eitc(annual_gross)
        disability = self.disability_benefits(monthly_gross)
        total = snap + med + ccdf + tanf + wic + head_start + eitc + disability["ssdi"] + disability["ssi"]
        return dict(
            snap=snap, medicaid=med, ccdf=ccdf, tanf=tanf,
            wic=wic, head_start=head_start, eitc=eitc,
            ssdi=disability["ssdi"], ssi=disability["ssi"],
            benefits_total=total,
        )


def monthly_gross(wage: float, hours_per_week: float) -> float:
    return wage * hours_per_week * WEEKS_PER_MONTH


def monthly_net_earnings(gross: float) -> float:
    after_payroll = gross * (1.0 - PAYROLL_TAX_RATE)
    after_income_tax = after_payroll * (1.0 - INCOME_TAX_RATE)
    return after_income_tax


def compute_bundle(wage: float, hours: float, location: LocationBenefits) -> dict:
    gross = monthly_gross(wage, hours)
    net_earn = monthly_net_earnings(gross)
    ben = location.total_benefits(gross)
    true_net_income = net_earn + ben["benefits_total"]
    return dict(
        monthly_gross=round(gross, 2),
        monthly_net_earnings=round(net_earn, 2),
        snap=round(ben["snap"], 2), medicaid=round(ben["medicaid"], 2),
        ccdf=round(ben["ccdf"], 2), tanf=round(ben["tanf"], 2),
        wic=round(ben["wic"], 2), head_start=round(ben["head_start"], 2),
        eitc=round(ben["eitc"], 2), ssdi=round(ben["ssdi"], 2),
        ssi=round(ben["ssi"], 2), benefits_total=round(ben["benefits_total"], 2),
        true_net_income=round(true_net_income, 2),
    )


def app_dir() -> Path:
    return Path(__file__).resolve().parent


def vignette_csv_path() -> Path:
    return app_dir() / "vignettes_60_strategic.csv"


def generate_vignettes_csv(n: int, seed: int) -> None:
    """Generate 60 vignettes: 20 CLIFF, 20 PLATEAU, 20 POSITIVE"""
    rng = random.Random(seed)

    locations = [
        ("PA", "Allegheny"), ("PA", "Philadelphia"),
        ("GA", "Fulton"), ("GA", "Cobb"),
        ("TX", "Harris"), ("TX", "Dallas"),
        ("NY", "New York"), ("NY", "Erie"),
        ("OH", "Cuyahoga"), ("OH", "Franklin"),
    ]

    family_types = [
        {"num_children": 0, "child_ages": [], "has_disability": False, "adult_age": 28, "is_married": False,
         "spouse_age": None, "other_adults": 0, "desc": "Single adult (age 28), no children, living alone"},
        {"num_children": 0, "child_ages": [], "has_disability": False, "adult_age": 35, "is_married": False,
         "spouse_age": None, "other_adults": 0, "desc": "Single adult (age 35), no children, living alone"},
        {"num_children": 0, "child_ages": [], "has_disability": False, "adult_age": 26, "is_married": False,
         "spouse_age": None, "other_adults": 1,
         "desc": "Single adult (age 26), living with 1 other adult (parent/sibling/roommate)"},
        {"num_children": 0, "child_ages": [], "has_disability": False, "adult_age": 31, "is_married": False,
         "spouse_age": None, "other_adults": 2, "desc": "Single adult (age 31), living with 2 other adults"},
        {"num_children": 0, "child_ages": [], "has_disability": False, "adult_age": 32, "is_married": True,
         "spouse_age": 34, "other_adults": 0, "desc": "Married couple (ages 32 & 34), no children"},
        {"num_children": 0, "child_ages": [], "has_disability": False, "adult_age": 42, "is_married": True,
         "spouse_age": 40, "other_adults": 0, "desc": "Married couple (ages 42 & 40), no children"},
        {"num_children": 1, "child_ages": [3], "has_disability": False, "adult_age": 32, "is_married": False,
         "spouse_age": None, "other_adults": 0, "desc": "Single parent (age 32), 1 child (age 3)"},
        {"num_children": 2, "child_ages": [2, 6], "has_disability": False, "adult_age": 34, "is_married": False,
         "spouse_age": None, "other_adults": 0, "desc": "Single parent (age 34), 2 children (ages 2, 6)"},
        {"num_children": 1, "child_ages": [7], "has_disability": False, "adult_age": 29, "is_married": False,
         "spouse_age": None, "other_adults": 1,
         "desc": "Single parent (age 29), 1 child (age 7), living with 1 other adult (grandparent)"},
        {"num_children": 2, "child_ages": [1, 4], "has_disability": False, "adult_age": 27, "is_married": False,
         "spouse_age": None, "other_adults": 1,
         "desc": "Single parent (age 27), 2 children (ages 1, 4), living with 1 other adult"},
        {"num_children": 1, "child_ages": [7], "has_disability": False, "adult_age": 38, "is_married": True,
         "spouse_age": 36, "other_adults": 0, "desc": "Married parents (ages 38 & 36), 1 child (age 7)"},
        {"num_children": 2, "child_ages": [4, 8], "has_disability": False, "adult_age": 36, "is_married": True,
         "spouse_age": 38, "other_adults": 0, "desc": "Married parents (ages 36 & 38), 2 children (ages 4, 8)"},
        {"num_children": 3, "child_ages": [2, 5, 10], "has_disability": False, "adult_age": 37, "is_married": True,
         "spouse_age": 39, "other_adults": 0, "desc": "Married parents (ages 37 & 39), 3 children (ages 2, 5, 10)"},
        {"num_children": 2, "child_ages": [3, 7], "has_disability": False, "adult_age": 35, "is_married": True,
         "spouse_age": 37, "other_adults": 1,
         "desc": "Married parents (ages 35 & 37), 2 children (ages 3, 7), living with 1 other adult (grandparent)"},
        {"num_children": 1, "child_ages": [14], "has_disability": False, "adult_age": 45, "is_married": False,
         "spouse_age": None, "other_adults": 0, "desc": "Single parent (age 45), 1 teen (age 14)"},
        {"num_children": 2, "child_ages": [13, 16], "has_disability": False, "adult_age": 47, "is_married": False,
         "spouse_age": None, "other_adults": 0, "desc": "Single parent (age 47), 2 teens (ages 13, 16)"},
        {"num_children": 2, "child_ages": [13, 16], "has_disability": False, "adult_age": 47, "is_married": True,
         "spouse_age": 49, "other_adults": 0, "desc": "Married parents (ages 47 & 49), 2 teens (ages 13, 16)"},
        {"num_children": 0, "child_ages": [], "has_disability": True, "adult_age": 33, "is_married": False,
         "spouse_age": None, "other_adults": 0, "desc": "Single adult with disability (age 33), living alone"},
        {"num_children": 0, "child_ages": [], "has_disability": True, "adult_age": 40, "is_married": True,
         "spouse_age": 42, "other_adults": 0, "desc": "Married couple (ages 40 & 42), primary earner has disability"},
        {"num_children": 1, "child_ages": [6], "has_disability": True, "adult_age": 36, "is_married": False,
         "spouse_age": None, "other_adults": 1,
         "desc": "Single parent with disability (age 36), 1 child (age 6), living with 1 other adult"},
        {"num_children": 2, "child_ages": [5, 9], "has_disability": True, "adult_age": 39, "is_married": True,
         "spouse_age": 41, "other_adults": 0,
         "desc": "Married parents (ages 39 & 41), 2 children (ages 5, 9), primary earner has disability"},
    ]

    cliff_scenarios = [
        {"current_wage": 11, "offer_wage": 14, "hours": 25},
        {"current_wage": 11, "offer_wage": 15, "hours": 30},
        {"current_wage": 12, "offer_wage": 16, "hours": 28},
        {"current_wage": 10, "offer_wage": 14, "hours": 32},
        {"current_wage": 11, "offer_wage": 15, "hours": 35},
        {"current_wage": 12, "offer_wage": 15, "hours": 30},
        {"current_wage": 11, "offer_wage": 14, "hours": 32},
        {"current_wage": 10, "offer_wage": 13, "hours": 35},
        {"current_wage": 12, "offer_wage": 16, "hours": 25},
        {"current_wage": 11, "offer_wage": 16, "hours": 28},
    ]

    plateau_scenarios = [
        {"current_wage": 13, "offer_wage": 14, "hours": 30},
        {"current_wage": 14, "offer_wage": 15, "hours": 32},
        {"current_wage": 15, "offer_wage": 16, "hours": 30},
        {"current_wage": 16, "offer_wage": 17, "hours": 28},
        {"current_wage": 14, "offer_wage": 15, "hours": 35},
        {"current_wage": 13, "offer_wage": 14, "hours": 32},
        {"current_wage": 15, "offer_wage": 16, "hours": 35},
        {"current_wage": 14, "offer_wage": 16, "hours": 30},
        {"current_wage": 16, "offer_wage": 17, "hours": 32},
        {"current_wage": 15, "offer_wage": 17, "hours": 28},
    ]

    positive_scenarios = [
        {"current_wage": 10, "offer_wage": 12, "hours": 20},
        {"current_wage": 11, "offer_wage": 13, "hours": 22},
        {"current_wage": 12, "offer_wage": 13, "hours": 25},
        {"current_wage": 10, "offer_wage": 11, "hours": 28},
        {"current_wage": 15, "offer_wage": 20, "hours": 35},
        {"current_wage": 16, "offer_wage": 22, "hours": 35},
        {"current_wage": 12, "offer_wage": 14, "hours": 22},
        {"current_wage": 11, "offer_wage": 12, "hours": 25},
        {"current_wage": 17, "offer_wage": 22, "hours": 40},
        {"current_wage": 14, "offer_wage": 20, "hours": 32},
    ]

    rows = []

    for scenario_type, scenarios, target_label in [
        ("CLIFF", cliff_scenarios, "cliff"),
        ("PLATEAU", plateau_scenarios, "plateau"),
        ("POSITIVE", positive_scenarios, "positive")
    ]:
        for iteration in range(2):
            for scenario_idx, scenario in enumerate(scenarios):
                state, county = rng.choice(locations)
                family = rng.choice(family_types)

                current_wage = float(scenario["current_wage"])
                offer_wage = float(scenario["offer_wage"])
                hours = scenario["hours"]

                current_wage += rng.uniform(-0.5, 0.5)
                offer_wage += rng.uniform(-0.5, 0.5)

                current_wage = max(10.0, current_wage)
                if offer_wage <= current_wage:
                    offer_wage = current_wage + rng.uniform(1.0, 2.0)

                location = LocationBenefits(
                    state=state, county=county,
                    num_children=family["num_children"],
                    child_ages=family["child_ages"],
                    has_disability=family["has_disability"]
                )

                cur = compute_bundle(current_wage, hours, location)
                off = compute_bundle(offer_wage, hours, location)

                delta_true = round(off["true_net_income"] - cur["true_net_income"], 2)
                delta_ben = round(off["benefits_total"] - cur["benefits_total"], 2)
                delta_netearn = round(off["monthly_net_earnings"] - cur["monthly_net_earnings"], 2)

                actual_type = "positive"
                if delta_netearn > 0 and delta_true < -50:
                    actual_type = "cliff"
                elif delta_netearn > 0 and -50 <= delta_true <= 100:
                    actual_type = "plateau"

                programs = ["snap", "medicaid", "ccdf", "tanf", "wic", "head_start", "eitc", "ssdi", "ssi"]
                programs_lost = sum([1 for prog in programs if cur[prog] > 0 and off[prog] == 0])
                programs_reduced = sum([1 for prog in programs if cur[prog] > off[prog] > 0])

                if family["num_children"] > 0:
                    child_ages_str = ", ".join(str(age) for age in family["child_ages"])
                else:
                    child_ages_str = "none"

                household_size = 1
                if family["is_married"]:
                    household_size += 1
                household_size += family["other_adults"]
                household_size += family["num_children"]

                row = dict(
                    vignette_id=len(rows) + 1,
                    scenario_type=actual_type,
                    target_type=target_label,
                    state=state, county=county,
                    family_description=family["desc"],
                    num_children=family["num_children"],
                    has_disability=int(family["has_disability"]),
                    adult_age=family["adult_age"],
                    is_married=int(family["is_married"]),
                    spouse_age=family["spouse_age"] if family["spouse_age"] is not None else 0,
                    other_adults=family["other_adults"],
                    household_size=household_size,
                    child_ages_str=child_ages_str,
                    current_hourly_wage=round(current_wage, 2),
                    current_hours=int(hours),
                    offer_hourly_wage=round(offer_wage, 2),
                    offer_hours=int(hours),
                    cur_true_net_income=cur["true_net_income"],
                    off_true_net_income=off["true_net_income"],
                    cur_monthly_gross=cur["monthly_gross"],
                    off_monthly_gross=off["monthly_gross"],
                    cur_monthly_net_earnings=cur["monthly_net_earnings"],
                    off_monthly_net_earnings=off["monthly_net_earnings"],
                    cur_snap=cur["snap"], off_snap=off["snap"],
                    cur_medicaid=cur["medicaid"], off_medicaid=off["medicaid"],
                    cur_ccdf=cur["ccdf"], off_ccdf=off["ccdf"],
                    cur_tanf=cur["tanf"], off_tanf=off["tanf"],
                    cur_wic=cur["wic"], off_wic=off["wic"],
                    cur_head_start=cur["head_start"], off_head_start=off["head_start"],
                    cur_eitc=cur["eitc"], off_eitc=off["eitc"],
                    cur_ssdi=cur["ssdi"], off_ssdi=off["ssdi"],
                    cur_ssi=cur["ssi"], off_ssi=off["ssi"],
                    cur_benefits_total=cur["benefits_total"],
                    off_benefits_total=off["benefits_total"],
                    delta_true_net_income=delta_true,
                    delta_benefits_total=delta_ben,
                    delta_net_earnings=delta_netearn,
                    is_cliff=int(delta_netearn > 0 and delta_true < 0),
                    programs_lost=programs_lost,
                    programs_reduced=programs_reduced,
                )
                rows.append(row)

    rng.shuffle(rows)

    for i, row in enumerate(rows):
        row["vignette_id"] = i + 1

    path = vignette_csv_path()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cliff_count = sum(1 for r in rows if r["scenario_type"] == "cliff")
    plateau_count = sum(1 for r in rows if r["scenario_type"] == "plateau")
    positive_count = sum(1 for r in rows if r["scenario_type"] == "positive")
    print(f"Generated: {cliff_count} CLIFF, {plateau_count} PLATEAU, {positive_count} POSITIVE")


def load_vignettes() -> list[dict]:
    path = vignette_csv_path()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k in {"vignette_id", "is_cliff", "programs_lost", "programs_reduced", "num_children",
                         "has_disability", "adult_age", "is_married", "spouse_age", "other_adults", "household_size"}:
                    parsed[k] = int(float(v))
                elif k in {"family_description", "state", "county", "scenario_type", "target_type", "child_ages_str"}:
                    parsed[k] = v
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            out.append(parsed)
    return out


# oTree models
class C(BaseConstants):
    NAME_IN_URL = "benefit_cliff"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    vignette_id = models.IntegerField()
    scenario_type = models.StringField()
    state = models.StringField()
    county = models.StringField()
    family_description = models.StringField()
    num_children = models.IntegerField()
    has_disability = models.IntegerField()
    adult_age = models.IntegerField()
    is_married = models.IntegerField()
    spouse_age = models.IntegerField()
    other_adults = models.IntegerField()
    household_size = models.IntegerField()
    child_ages_str = models.StringField()

    current_hourly_wage = models.FloatField()
    current_hours = models.IntegerField()
    offer_hourly_wage = models.FloatField()
    offer_hours = models.IntegerField()

    cur_snap = models.FloatField()
    cur_medicaid = models.FloatField()
    cur_ccdf = models.FloatField()
    cur_tanf = models.FloatField()
    cur_wic = models.FloatField()
    cur_head_start = models.FloatField()
    cur_eitc = models.FloatField()
    cur_ssdi = models.FloatField()
    cur_ssi = models.FloatField()
    cur_benefits_total = models.FloatField()
    cur_true_net_income = models.FloatField()

    off_snap = models.FloatField()
    off_medicaid = models.FloatField()
    off_ccdf = models.FloatField()
    off_tanf = models.FloatField()
    off_wic = models.FloatField()
    off_head_start = models.FloatField()
    off_eitc = models.FloatField()
    off_ssdi = models.FloatField()
    off_ssi = models.FloatField()
    off_benefits_total = models.FloatField()
    off_true_net_income = models.FloatField()

    delta_true_net_income = models.FloatField()
    delta_net_earnings = models.FloatField()
    delta_benefits_total = models.FloatField()
    is_cliff = models.IntegerField()
    programs_lost = models.IntegerField()
    programs_reduced = models.IntegerField()

    accept_offer = models.IntegerField(
        choices=[[1, "Accept the offer"], [0, "Keep current job"]],
        widget=widgets.RadioSelect,
        label="What would you do?"
    )

    perceived_delta_true_net_income = models.FloatField(
        label="What do you think the change in your monthly total resources would be?",
        blank=False,
        min=-10000,
        max=10000,
    )


def creating_session(subsession: Subsession):
    if not vignette_csv_path().exists():
        generate_vignettes_csv(N_VIGNETTES, SEED)

    vignettes = load_vignettes()
    players = subsession.get_players()
    rng = random.Random(SEED + subsession.session.config.get('random_seed', 0))

    n_players = len(players)
    repetitions_per_vignette = (n_players // len(vignettes)) + 1

    assignments = []
    for _ in range(repetitions_per_vignette):
        assignments.extend(vignettes)

    rng.shuffle(assignments)
    assignments = assignments[:n_players]

    for p, v in zip(players, assignments[:n_players]):
        p.vignette_id = v["vignette_id"]
        p.scenario_type = v["scenario_type"]
        p.state = v["state"]
        p.county = v["county"]
        p.family_description = v["family_description"]
        p.num_children = v["num_children"]
        p.has_disability = v["has_disability"]
        p.adult_age = v["adult_age"]
        p.is_married = v["is_married"]
        p.spouse_age = v["spouse_age"]
        p.other_adults = v["other_adults"]
        p.household_size = v["household_size"]
        p.child_ages_str = v["child_ages_str"]
        p.current_hourly_wage = v["current_hourly_wage"]
        p.current_hours = v["current_hours"]
        p.offer_hourly_wage = v["offer_hourly_wage"]
        p.offer_hours = v["offer_hours"]

        p.cur_snap = v["cur_snap"]
        p.cur_medicaid = v["cur_medicaid"]
        p.cur_ccdf = v["cur_ccdf"]
        p.cur_tanf = v["cur_tanf"]
        p.cur_wic = v["cur_wic"]
        p.cur_head_start = v["cur_head_start"]
        p.cur_eitc = v["cur_eitc"]
        p.cur_ssdi = v["cur_ssdi"]
        p.cur_ssi = v["cur_ssi"]
        p.cur_benefits_total = v["cur_benefits_total"]
        p.cur_true_net_income = v["cur_true_net_income"]

        p.off_snap = v["off_snap"]
        p.off_medicaid = v["off_medicaid"]
        p.off_ccdf = v["off_ccdf"]
        p.off_tanf = v["off_tanf"]
        p.off_wic = v["off_wic"]
        p.off_head_start = v["off_head_start"]
        p.off_eitc = v["off_eitc"]
        p.off_ssdi = v["off_ssdi"]
        p.off_ssi = v["off_ssi"]
        p.off_benefits_total = v["off_benefits_total"]
        p.off_true_net_income = v["off_true_net_income"]

        p.delta_true_net_income = v["delta_true_net_income"]
        p.delta_net_earnings = v["delta_net_earnings"]
        p.delta_benefits_total = v["delta_benefits_total"]
        p.is_cliff = v["is_cliff"]
        p.programs_lost = v["programs_lost"]
        p.programs_reduced = v["programs_reduced"]


# ============================================
# PAGES
# ============================================

# NEW PAGE 1: Welcome from Qualtrics
class WelcomeFromQualtrics(Page):
    """First page after participants arrive from Qualtrics"""

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

    @staticmethod
    def before_next_page(player, timeout_happened):
        participant = player.participant
        session = player.session

        # Capture IDs from URL parameters
        prolific_pid = session.config.get('prolific_pid', 'TEST_NO_ID')
        qualtrics_id = session.config.get('qualtrics_id', 'TEST_NO_ID')

        # Store in participant fields
        participant.prolific_pid = prolific_pid
        participant.qualtrics_response_id = qualtrics_id

        # Generate completion code for Prolific
        completion_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        participant.completion_code = completion_code

        # Debug print
        print(f"=== Participant from Qualtrics ===")
        print(f"Prolific PID: {prolific_pid}")
        print(f"Qualtrics Response ID: {qualtrics_id}")
        print(f"Generated Completion Code: {completion_code}")


# EXISTING PAGE 2: Main decision page
class MyPage(Page):
    form_model = "player"
    form_fields = ["accept_offer", "perceived_delta_true_net_income"]

    @staticmethod
    def vars_for_template(player: Player):
        cur_gross = player.current_hourly_wage * player.current_hours * WEEKS_PER_MONTH
        off_gross = player.offer_hourly_wage * player.offer_hours * WEEKS_PER_MONTH

        cur_annual = cur_gross * 12
        off_annual = off_gross * 12

        show_ccdf = player.num_children > 0
        show_tanf = player.num_children > 0
        show_wic = player.num_children > 0
        show_head_start = player.num_children > 0
        show_disability = player.has_disability == 1

        return dict(
            current_wage=player.current_hourly_wage,
            current_hours=player.current_hours,
            current_monthly_gross=round(cur_gross, 2),
            current_annual_gross=round(cur_annual, 2),
            offer_monthly_gross=round(off_gross, 2),
            offer_annual_gross=round(off_annual, 2),
            show_benefits=True,
            show_ccdf=show_ccdf,
            show_tanf=show_tanf,
            show_wic=show_wic,
            show_head_start=show_head_start,
            show_disability=show_disability,
        )


# EXISTING PAGE 3: Results page
class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        perception_error = None
        if player.perceived_delta_true_net_income is not None:
            perception_error = round(
                player.perceived_delta_true_net_income - player.delta_true_net_income,
                2
            )

        show_ccdf = player.num_children > 0
        show_tanf = player.num_children > 0
        show_wic = player.num_children > 0
        show_head_start = player.num_children > 0
        show_disability = player.has_disability == 1

        return dict(
            show_truth=True,
            perception_error=perception_error,
            show_ccdf=show_ccdf,
            show_tanf=show_tanf,
            show_wic=show_wic,
            show_head_start=show_head_start,
            show_disability=show_disability,
        )


# NEW PAGE 4: Final page with Prolific completion code
class FinalPageProlific(Page):
    """Final page with Prolific completion code"""

    @staticmethod
    def is_displayed(player):
        return player.round_number == C.NUM_ROUNDS

    @staticmethod
    def vars_for_template(player):
        completion_code = player.participant.completion_code
        return {
            'completion_code': completion_code,
            'prolific_completion_url': f'https://app.prolific.com/submissions/complete?cc={completion_code}'
        }


# PAGE SEQUENCE - Updated with new pages
page_sequence = [
    WelcomeFromQualtrics,  # NEW - First
    MyPage,  # Existing
    Results,  # Existing
    FinalPageProlific,  # NEW - Last
]