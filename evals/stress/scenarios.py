"""Escenarios multi-turno del stress test del CAG.

Cada escenario son >= 20 turnos coherentes sobre el mismo proyecto más un fact-tracker: hechos
literales (búsqueda de subcadena, sin LLM) que se declaran en un turno y se comprueban en los
siguientes con `MemoryDriftMetric`. Los textos evitan a propósito lo que activaría otras piezas
del sistema sin ser el objeto de la medida: PII, patrones de prompt injection, palabras de NDA o
de regulación (tier) y reglas de ancla, salvo las anclas deliberadas del escenario `contradiction`.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fact:
    """Hecho verificable por subcadena. `superseded_at`: turno que lo reemplaza (None = vigente)."""

    text: str
    declared_at: int  # 1-based
    superseded_at: int | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    turns: tuple[str, ...]
    facts: tuple[Fact, ...] = field(default_factory=tuple)

    def expected_facts(self, turn_index: int) -> list[str]:
        """Hechos que la memoria debería conservar tras el turno `turn_index`."""
        return [
            f.text
            for f in self.facts
            if f.declared_at <= turn_index and (f.superseded_at is None or turn_index < f.superseded_at)
        ]

    def stale_facts(self, turn_index: int) -> list[str]:
        """Hechos ya reemplazados: si siguen en la memoria, ésta arrastra información obsoleta."""
        return [f.text for f in self.facts if f.superseded_at is not None and f.superseded_at <= turn_index]


GROWING = Scenario(
    name="growing",
    description="Requisitos que se añaden poco a poco: contexto y coste acumulados, y supervivencia del nombre.",
    turns=(
        "We are building Northwind Booking, a web SaaS that lets boutique gyms sell class bookings online.",
        "The first release needs a public class schedule where members can browse and reserve a spot.",
        "The backend will use PostgreSQL to store members, classes and reservations.",
        "Add a waitlist: when a class is full, members join a queue and get promoted automatically.",
        "Members should be able to cancel a reservation up to two hours before the class starts.",
        "Payments will go through Stripe, with monthly memberships and single class passes.",
        "We also need an admin dashboard where gym owners manage instructors and weekly timetables.",
        "Instructors need a mobile friendly view of their own classes and attendee lists.",
        "Login should use Auth0 so members can sign in with Google or with a password.",
        "Add email reminders the evening before each reserved class.",
        "Owners want a report with attendance per class and revenue per month.",
        "Text message reminders will be sent through Twilio for members who opt in.",
        "Add support for multiple gym locations under the same owner account.",
        "Each location must have its own timetable, rooms and capacity limits.",
        "Members want a map of nearby locations, we plan to use Mapbox for it.",
        "Add a referral program where a member gets a free class after a friend joins.",
        "Owners should be able to export attendance data as spreadsheet files.",
        "We need an audit log of every change made to timetables and prices.",
        "Performance goal: the schedule page must load in under two seconds on a mobile connection.",
        "The hosting will be on AWS, using managed PostgreSQL and a CDN for static assets.",
        "Add a simple in app messaging feature between members and instructors.",
        "Please give the updated estimate for everything we have discussed about Northwind Booking so far.",
    ),
    facts=(
        Fact("Northwind Booking", 1),
        Fact("PostgreSQL", 3),
        Fact("Stripe", 6),
        Fact("Auth0", 9),
        Fact("Twilio", 12),
        Fact("Mapbox", 15),
        Fact("AWS", 20),
    ),
)

PIVOT = Scenario(
    name="pivot",
    description="Cambio de tecnología en el turno 5: ¿la metadata sustituye el stack anterior o acumula ambos?",
    turns=(
        "We are building Atlas Ledger, a web SaaS that helps small accounting firms reconcile invoices.",
        "The backend will be written in Django with Django REST framework.",
        "Users upload bank statements and the system matches each line with an open invoice.",
        "Matching should suggest the best candidate and let the accountant confirm it with one click.",
        "Big change: we are dropping Django and the backend will now be built with NestJS on Node.js.",
        "With NestJS we will expose a REST API and generate the documentation from decorators.",
        "The team already knows TypeScript, so the web client will also be written in TypeScript.",
        "Add a review queue for matches the system is not confident about.",
        "Accountants need a monthly close checklist that shows pending reconciliations.",
        "Every confirmed match must record who confirmed it and when.",
        "We want to import invoices from a shared folder every night.",
        "Add multi currency support with daily exchange rates.",
        "Owners need a dashboard with the reconciliation progress per client.",
        "Add role based access so that interns can only view, not confirm matches.",
        "The matching engine must handle about fifty thousand bank lines per month per firm.",
        "Add a rules editor where accountants define automatic matching rules for repeated payments.",
        "We need an undo feature for confirmed matches within the monthly close period.",
        "Add notifications when a bank statement upload fails or has unreadable lines.",
        "Include an onboarding wizard that imports the first set of invoices.",
        "Prepare the deployment as a single containerized service plus a managed database.",
        "Add a basic public API so firms can integrate Atlas Ledger with their own tools.",
        "Please give the updated estimate for everything we have discussed about Atlas Ledger so far.",
    ),
    facts=(
        Fact("Atlas Ledger", 1),
        Fact("Django", 2, superseded_at=5),
        Fact("NestJS", 5),
    ),
)

# Literales sin separador de miles ("30000"): el LLM suele conservarlos tal cual en metadata y resumen
CONTRADICTION = Scenario(
    name="contradiction",
    description="Presupuesto de 30000 EUR contradicho por 80000 EUR en el turno 8: qué persiste y dónde.",
    turns=(
        "We are building Harvest Hub, a web SaaS marketplace where local farmers sell produce to restaurants.",
        "Important: the budget is locked at 30000 EUR for the whole project.",
        "Farmers publish weekly availability lists with prices per kilo.",
        "Restaurants browse the lists and place standing orders for the week.",
        "Delivery is handled by the farmers, so we only need to track the delivery window.",
        "Add a rating system where restaurants rate each delivery.",
        "We need invoices generated automatically at the end of each week.",
        "The investors changed the plan: the total budget is now 80000 EUR.",
        "With the larger budget we can add a mobile app for farmers to update availability.",
        "Restaurants want to repeat last week's order with one click.",
        "Add a seasonal calendar that shows which products are in season.",
        "We need basic analytics on the most requested products per region.",
        "Add a messaging thread between a farmer and a restaurant for each order.",
        "Support payments by bank transfer and by card.",
        "Farmers want a way to mark a product as sold out in real time.",
        "Add a simple moderation queue for new farmer accounts before they can publish.",
        "Restaurants need a monthly statement with all their orders and invoices.",
        "Add an option for restaurants to share an account between several kitchen managers.",
        "We want to prepare a pilot with ten farmers and twenty restaurants in one region.",
        "The pilot should run for three months before the national launch.",
        "Add a public landing page that explains how Harvest Hub works.",
        "Please give the updated estimate for Harvest Hub and remind me what budget we are working with.",
    ),
    facts=(
        Fact("Harvest Hub", 1),
        Fact("30000", 2, superseded_at=8),
        Fact("80000", 8),
    ),
)

SCENARIOS: dict[str, Scenario] = {s.name: s for s in (GROWING, PIVOT, CONTRADICTION)}
