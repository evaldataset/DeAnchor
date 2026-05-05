"""실제 SEC 제재 사례 패턴 기반 사기 사례 생성.

SEC/FinCEN 실제 사례를 참고하여 구조화된 사기 사례 데이터 생성.
RAG 시스템 개발/테스트용.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cases" / "sec_enforcement"

# 실제 SEC/FinCEN 제재 사례 패턴 기반 사례 데이터
SAMPLE_CASES = [
    {
        "id": "SEC-0001",
        "source": "SEC Litigation Release",
        "title": "SEC Charges Investment Adviser with Structuring Cash Deposits to Evade Reporting",
        "year": "2023",
        "content": """The Securities and Exchange Commission charged a registered investment adviser
with structuring cash deposits at multiple bank branches to evade Currency Transaction Report (CTR)
requirements. Between January 2020 and March 2022, the adviser made over 200 cash deposits, each
between $8,000 and $9,900, at various branches of two national banks. The total amount deposited
exceeded $1.8 million. The deposits were made in a pattern designed to avoid the $10,000 CTR
threshold. Multiple deposits were made on the same day at different branches. The adviser used
client funds for personal expenses and attempted to conceal the misappropriation through these
structured deposits. The SEC obtained a court order freezing the adviser's assets and is seeking
disgorgement, penalties, and a permanent injunction. Violation of Section 17(a) of the Securities
Act and Section 10(b) of the Exchange Act.""",
        "metadata": {
            "fraud_types": ["Structuring"],
            "amounts_mentioned": ["$8,000", "$9,900", "$1.8 million", "$10,000"],
            "regulations": ["Section 17(a) Securities Act", "Section 10(b) Exchange Act", "31 USC 5324"],
        },
    },
    {
        "id": "SEC-0002",
        "source": "SEC Litigation Release",
        "title": "SEC Charges Broker-Dealer with Layering Scheme in Equity Markets",
        "year": "2023",
        "content": """The SEC filed charges against a broker-dealer and two traders for engaging in
a layering scheme that generated approximately $3.2 million in illicit profits. The traders placed
large non-bona fide orders on one side of the market to create the appearance of supply or demand,
then executed smaller orders on the opposite side at artificial prices, and promptly cancelled the
non-bona fide orders. The scheme involved over 675,000 instances of layering across 3,500 securities
over a two-year period. The traders used multiple accounts and algorithmic tools to execute the
strategy at high speed. The broker-dealer failed to establish adequate supervisory systems to detect
the manipulative activity despite red flags including unusually high order cancellation rates
exceeding 95% and rapid order placement and cancellation patterns. The SEC seeks permanent
injunctions, disgorgement of ill-gotten gains with interest, and civil penalties.""",
        "metadata": {
            "fraud_types": ["Layering", "Market Manipulation"],
            "amounts_mentioned": ["$3.2 million"],
            "regulations": ["Section 9(a)(2) Exchange Act", "Section 10(b) Exchange Act", "Rule 10b-5"],
        },
    },
    {
        "id": "SEC-0003",
        "source": "SEC Litigation Release",
        "title": "SEC Obtains Final Judgment Against Identity Fraud Ring Targeting Brokerage Accounts",
        "year": "2022",
        "content": """The SEC obtained a final judgment against members of an identity fraud ring
that stole personal information to open fraudulent brokerage accounts and execute unauthorized
trades. The defendants obtained Social Security numbers, dates of birth, and other personal
information through data breaches and social engineering. They opened accounts at multiple
broker-dealers using synthetic identities combining real and fabricated information. Once accounts
were funded through ACH transfers from compromised bank accounts, the defendants executed a series
of rapid trades in thinly-traded securities to generate profits, which were then withdrawn through
wire transfers to accounts controlled by the ring. Total losses exceeded $4.7 million affecting
over 200 victims. The scheme operated across state lines using VPNs and prepaid devices to obscure
the perpetrators' locations.""",
        "metadata": {
            "fraud_types": ["Identity Fraud", "Account Takeover"],
            "amounts_mentioned": ["$4.7 million"],
            "regulations": ["Section 17(a) Securities Act", "Section 10(b) Exchange Act"],
        },
    },
    {
        "id": "SEC-0004",
        "source": "SEC Litigation Release",
        "title": "Corporate Executive Charged with Insider Trading Ahead of Merger Announcement",
        "year": "2023",
        "content": """The SEC charged a senior vice president of a publicly traded technology
company with insider trading based on material, non-public information about an upcoming merger.
The executive learned about the pending $2.8 billion acquisition during a confidential board
meeting. In the weeks before the public announcement, the executive purchased 15,000 shares and
call options worth approximately $340,000 in the target company through a brokerage account held
in a family member's name. After the merger was publicly announced, the stock price increased 47%,
and the executive realized profits exceeding $620,000. Trading records showed the executive had
no prior history of trading the target company's securities. The timing of purchases closely
correlated with access to confidential merger documents. Phone records showed communications
between the executive and the family member immediately before each trade.""",
        "metadata": {
            "fraud_types": ["Insider Trading"],
            "amounts_mentioned": ["$2.8 billion", "$340,000", "$620,000"],
            "regulations": ["Section 10(b) Exchange Act", "Rule 10b-5", "Section 14(e) Exchange Act"],
        },
    },
    {
        "id": "SEC-0005",
        "source": "SEC Litigation Release",
        "title": "SEC Halts Pump-and-Dump Scheme Promoted Through Social Media",
        "year": "2024",
        "content": """The SEC obtained an emergency court order halting a pump-and-dump scheme
involving micro-cap securities promoted through coordinated social media campaigns on multiple
platforms. The defendants acquired large positions in three penny stocks at prices below $0.10
per share, then orchestrated promotional campaigns using paid influencers and fake accounts to
tout the stocks as having breakthrough technology. The social media posts contained materially
false and misleading claims about partnerships, revenue projections, and regulatory approvals.
As the stock prices were artificially inflated by up to 800%, the defendants sold their positions
for combined profits exceeding $8.5 million. Trading volume in the affected securities increased
from an average of 50,000 shares per day to over 15 million shares during the promotional period.
The SEC's investigation revealed that the defendants controlled over 250 social media accounts
used in the scheme.""",
        "metadata": {
            "fraud_types": ["Pump & Dump", "Market Manipulation"],
            "amounts_mentioned": ["$0.10", "$8.5 million"],
            "regulations": ["Section 17(a) Securities Act", "Section 10(b) Exchange Act", "Rule 10b-5"],
        },
    },
    {
        "id": "SEC-0006",
        "source": "FinCEN Enforcement Action",
        "title": "FinCEN Assesses Civil Penalty Against Bank for BSA Violations in Structuring Detection",
        "year": "2023",
        "content": """FinCEN assessed a $15 million civil money penalty against a regional bank
for willful violations of the Bank Secrecy Act. The bank failed to implement adequate anti-money
laundering controls, resulting in the failure to detect and report thousands of suspicious
transactions over a five-year period. The bank's transaction monitoring system had significant
gaps, including thresholds set too high to capture structuring activity, insufficient monitoring
of cash-intensive business accounts, and failure to aggregate related transactions across
branches. During the examination period, the bank processed over $200 million in suspicious
transactions without filing Suspicious Activity Reports. The failures were particularly acute
in the bank's foreign correspondent banking relationships, where transaction patterns consistent
with layering and trade-based money laundering went undetected. The bank has agreed to an enhanced
compliance program and independent monitor.""",
        "metadata": {
            "fraud_types": ["Structuring", "Money Laundering", "Layering"],
            "amounts_mentioned": ["$15 million", "$200 million", "$10,000"],
            "regulations": ["31 USC 5318", "31 CFR 1020.320", "Bank Secrecy Act"],
        },
    },
    {
        "id": "SEC-0007",
        "source": "SEC Litigation Release",
        "title": "Former Bank Employee Charged with Account Takeover and Unauthorized Transfers",
        "year": "2022",
        "content": """The SEC and DOJ charged a former bank employee with conducting unauthorized
transfers from customer accounts to accounts controlled by co-conspirators. Over an eighteen-month
period, the employee accessed customer accounts using privileged system credentials and initiated
wire transfers totaling $2.3 million. The transfers were structured to avoid triggering automated
alerts: amounts were kept below $5,000, transfers were spread across different days and times, and
recipient accounts were changed frequently. The employee also modified account contact information
to prevent customers from receiving transfer notifications. The scheme was discovered when a
customer noticed discrepancies in monthly statements and reported the unauthorized activity.
Investigation revealed that the employee had accessed over 150 customer accounts without
authorization.""",
        "metadata": {
            "fraud_types": ["Account Takeover", "Insider Trading"],
            "amounts_mentioned": ["$2.3 million", "$5,000"],
            "regulations": ["18 USC 1343 Wire Fraud", "18 USC 1344 Bank Fraud"],
        },
    },
    {
        "id": "SEC-0008",
        "source": "SEC Litigation Release",
        "title": "International Money Laundering Network Using Shell Companies and Crypto Exchanges",
        "year": "2024",
        "content": """The SEC, in coordination with international law enforcement, disrupted a
money laundering network that processed over $50 million through a web of shell companies and
cryptocurrency exchanges. The network operated by receiving proceeds from various predicate
offenses including ransomware attacks, romance scams, and investment fraud. Funds were layered
through multiple bank accounts held by shell companies registered in different jurisdictions.
The shell companies had no legitimate business operations but maintained the appearance of
active businesses through fabricated invoices and contracts. Funds were then converted to
cryptocurrency through multiple exchanges, tumbled through privacy protocols, and converted
back to fiat currency through over-the-counter brokers. The network used a combination of
traditional banking and cryptocurrency systems to maximize complexity and minimize traceability.
Each layer of transactions was designed to separate the funds from their illegal origin by at
least three steps.""",
        "metadata": {
            "fraud_types": ["Money Laundering", "Layering"],
            "amounts_mentioned": ["$50 million"],
            "regulations": ["18 USC 1956 Money Laundering", "31 USC 5311 BSA"],
        },
    },
    {
        "id": "SEC-0009",
        "source": "SEC Litigation Release",
        "title": "SEC Charges Fund Manager with Misappropriation Through Round-Trip Transactions",
        "year": "2023",
        "content": """The SEC charged a hedge fund manager with misappropriating approximately
$12 million in fund assets through a series of round-trip transactions designed to appear as
legitimate investment activity. The manager created fictitious trades between the fund and
entities he secretly controlled, generating the appearance of trading profits while actually
siphoning fund capital. Monthly statements sent to investors showed consistent positive returns
of 2-3% per month, but the underlying trades were fabricated. The manager used the misappropriated
funds to support a lavish personal lifestyle, including luxury real estate, vehicles, and travel.
When investor redemption requests increased, the manager engaged in a Ponzi-like scheme, using
new investor capital to satisfy existing redemption requests. The fraud was discovered during a
routine SEC examination when auditors could not verify the counterparties to reported trades.""",
        "metadata": {
            "fraud_types": ["Ponzi Scheme", "Layering"],
            "amounts_mentioned": ["$12 million"],
            "regulations": ["Section 206 Investment Advisers Act", "Section 17(a) Securities Act"],
        },
    },
    {
        "id": "SEC-0010",
        "source": "SEC Litigation Release",
        "title": "Credit Card Fraud Ring Using Synthetic Identities for Cash Advances",
        "year": "2023",
        "content": """Federal prosecutors charged members of a fraud ring that used synthetic
identities to obtain credit cards and extract over $6.2 million through cash advances and
purchases. The ring created hundreds of synthetic identities by combining real Social Security
numbers belonging to minors, elderly individuals, and deceased persons with fabricated names
and addresses. The identities were used to apply for credit cards at multiple financial
institutions. Once approved, the cards were used for rapid cash advances at ATMs and purchases
of high-value items that were immediately resold. Each synthetic identity was maintained for
6-12 months, during which the ring made regular small payments to build credit history before
executing the cash-out phase. The scheme involved coordination across multiple states, with
different members responsible for identity creation, credit building, and cash extraction.
The ring used encrypted communications and prepaid phones to coordinate activities.""",
        "metadata": {
            "fraud_types": ["Identity Fraud", "Structuring"],
            "amounts_mentioned": ["$6.2 million"],
            "regulations": ["18 USC 1029 Fraud with Access Devices", "18 USC 1028 Identity Fraud"],
        },
    },
    {
        "id": "SEC-0011",
        "source": "SEC Litigation Release",
        "title": "Legitimate Business Transaction Cleared After Investigation of Large Wire Transfer",
        "year": "2023",
        "content": """A compliance investigation was triggered by a series of large wire transfers
totaling $890,000 from a small business account. The account holder, a commercial real estate
company, made three wire transfers of $280,000, $310,000, and $300,000 within a five-day period
to a previously unknown beneficiary account. Initial analysis flagged the transactions due to:
deviation from historical transaction patterns, large total amount, transfers to a new beneficiary,
and velocity of transactions. Upon investigation, the transactions were determined to be legitimate
earnest money deposits for a commercial property acquisition. Supporting documentation included a
signed purchase agreement, escrow instructions from a licensed title company, and correspondence
between real estate attorneys. The business had an established history of similar transactions
during previous property acquisitions, though at a lower frequency. The investigation was closed
with no SAR filing after verification of the underlying commercial transaction.""",
        "metadata": {
            "fraud_types": ["Legitimate"],
            "amounts_mentioned": ["$890,000", "$280,000", "$310,000", "$300,000"],
            "regulations": [],
        },
    },
    {
        "id": "SEC-0012",
        "source": "SEC Litigation Release",
        "title": "Legitimate Payroll Processing Flagged Due to Round-Number Transfers",
        "year": "2022",
        "content": """An automated monitoring system flagged a series of weekly wire transfers
of exactly $9,500 from a staffing agency's operating account to multiple individual accounts.
The pattern was flagged due to: consistent round amounts just below the $10,000 CTR threshold,
regular timing, and multiple recipients. Investigation revealed the transfers were legitimate
payroll disbursements for temporary workers. The staffing agency's standard weekly pay for
full-time temporary placements was $9,500 (40 hours at $237.50/hour, a common billing rate for
specialized IT contractors). The agency provided employment contracts, timesheets verified by
client companies, and tax documentation (W-2s and 1099s) for all recipients. The regular amount
was a function of standardized billing rates rather than an attempt to avoid reporting thresholds.
No suspicious activity was identified.""",
        "metadata": {
            "fraud_types": ["Legitimate"],
            "amounts_mentioned": ["$9,500", "$10,000", "$237.50"],
            "regulations": [],
        },
    },
    # --- Structuring cases (SEC-0013 to SEC-0017) ---
    {
        "id": "SEC-0013",
        "source": "FinCEN Enforcement Action",
        "title": "Restaurant Chain Owner Charged with Structuring Cash Deposits from Business Proceeds",
        "year": "2024",
        "content": """FinCEN and the IRS charged the owner of a chain of fourteen restaurants with
structuring cash deposits over a three-year period. The owner directed managers at each restaurant
location to make daily cash deposits of between $7,500 and $9,800 at nearby bank branches rather
than depositing the full day's cash receipts in a single transaction. Internal records showed that
several locations regularly generated daily cash receipts exceeding $15,000. When aggregate daily
receipts exceeded $10,000, the owner instructed managers to split deposits across two or more
branches or hold a portion of cash for deposit the following day. In total, the owner structured
approximately $4.2 million in cash deposits across 612 separate transactions over 36 months. Bank
records showed a conspicuous absence of any single deposit at or above $10,000 despite the high
cash volume of the business. The owner also maintained multiple accounts at three different banks
to further distribute the deposit activity. The IRS Criminal Investigation division determined that
the structured deposits were used to conceal unreported income, resulting in additional charges of
tax evasion. The defendant faces up to five years imprisonment for each structuring count and ten
years for tax evasion.""",
        "metadata": {
            "fraud_types": ["Structuring"],
            "amounts_mentioned": ["$7,500", "$9,800", "$15,000", "$4.2 million", "$10,000"],
            "regulations": ["31 USC 5324", "26 USC 7201 Tax Evasion"],
        },
    },
    {
        "id": "SEC-0014",
        "source": "DOJ Criminal Complaint",
        "title": "Smurfing Network Charged with Structuring Over $9 Million Through Multiple Banks",
        "year": "2023",
        "content": """The Department of Justice charged eleven individuals with operating a smurfing
network that structured over $9.3 million in cash deposits through accounts at twenty-two different
financial institutions. The network recruited individuals, referred to as smurfs, through social
media advertisements offering payment for simple bank transactions. Each smurf was given cash in
amounts between $3,000 and $9,000 and instructed to deposit the funds into designated accounts at
specific bank branches. Smurfs were rotated among branches on a weekly basis to avoid recognition
by tellers. The network leader coordinated activities through encrypted messaging applications,
providing daily assignments specifying amounts, bank branches, and account numbers. Each smurf
completed three to five deposits per day across different financial institutions. The deposited
funds were promptly transferred via wire to accounts in Mexico and Colombia. Law enforcement
identified the network through patterns of related deposits detected by FinCEN's BSA data
analysis. Surveillance footage confirmed multiple smurfs visiting the same branches on rotating
schedules. Seventeen bank accounts were seized containing approximately $1.4 million in remaining
funds.""",
        "metadata": {
            "fraud_types": ["Structuring", "Money Laundering"],
            "amounts_mentioned": ["$9.3 million", "$3,000", "$9,000", "$1.4 million"],
            "regulations": ["31 USC 5324", "18 USC 1956 Money Laundering"],
        },
    },
    {
        "id": "SEC-0015",
        "source": "FinCEN Enforcement Action",
        "title": "Used Car Dealership Penalized for Structuring Wire Transfers to Avoid Reporting",
        "year": "2024",
        "content": """FinCEN assessed a civil money penalty of $2.1 million against a used car
dealership and its principal for structuring outgoing wire transfers to avoid Currency Transaction
Report filings. The dealership purchased vehicles at wholesale auctions and paid suppliers through
wire transfers deliberately kept below $10,000. Over an eighteen-month period, the dealership
initiated 847 wire transfers averaging $8,200 each, totaling approximately $6.9 million. On
numerous occasions, the dealership split payments for a single vehicle purchase into two or three
separate wire transfers on consecutive days. For example, a $22,000 vehicle purchase was paid
through three wires of $7,500, $7,500, and $7,000 sent over three days. The dealership's principal
admitted to structuring the transactions after receiving advice from an unlicensed financial
consultant who suggested the practice would reduce regulatory scrutiny. The bank handling the
dealership's account also received a separate penalty of $500,000 for failing to detect the
obvious structuring pattern despite automated alerts. FinCEN noted that the dealership is also
subject to its own anti-money laundering obligations as a dealer in vehicles.""",
        "metadata": {
            "fraud_types": ["Structuring"],
            "amounts_mentioned": ["$2.1 million", "$10,000", "$8,200", "$6.9 million", "$22,000", "$7,500", "$7,000", "$500,000"],
            "regulations": ["31 USC 5324", "31 CFR 1020.220", "Bank Secrecy Act"],
        },
    },
    {
        "id": "SEC-0016",
        "source": "DOJ Criminal Complaint",
        "title": "Foreign National Convicted of Structuring Cash Withdrawals to Fund Human Smuggling",
        "year": "2022",
        "content": """A federal jury convicted a foreign national of structuring cash withdrawals
from bank accounts to fund a human smuggling operation. The defendant made 340 cash withdrawals
over a fourteen-month period from accounts at four different banks. Each withdrawal was between
$4,000 and $9,500, and multiple withdrawals were made on the same day from different branches.
Total withdrawals exceeded $2.6 million. The cash was used to pay smuggling fees, safe house
operators, and transportation costs along the southwest border. The defendant maintained account
balances through regular deposits of checks from a landscaping business that served as a front.
The landscaping business reported minimal revenue on tax returns but processed deposits far
exceeding its declared income. Bank tellers at two branches reported the defendant's frequent
withdrawal pattern through internal suspicious activity referrals, but the bank's compliance
department failed to file SARs in a timely manner. The investigation was initiated by Homeland
Security Investigations after an informant identified the defendant as a financier for the
smuggling network. The defendant was sentenced to 48 months in federal prison.""",
        "metadata": {
            "fraud_types": ["Structuring"],
            "amounts_mentioned": ["$4,000", "$9,500", "$2.6 million"],
            "regulations": ["31 USC 5324", "8 USC 1324 Human Smuggling"],
        },
    },
    {
        "id": "SEC-0017",
        "source": "FinCEN Enforcement Action",
        "title": "Check Cashing Business Penalized for Systematic Structuring of Deposits",
        "year": "2023",
        "content": """FinCEN imposed a $3.8 million civil penalty on a licensed check cashing
business for systematically structuring currency deposits to evade BSA reporting requirements.
The business cashed checks for customers and accumulated significant cash reserves daily. Rather
than depositing its full daily cash holdings, the business owner directed employees to make
multiple deposits at different bank branches, each under $10,000. Over a four-year period, the
business made over 2,800 structured deposits totaling approximately $22 million. On days when
cash on hand exceeded $30,000, deposits were split among as many as four different branches.
The business maintained accounts at six different banks to facilitate the structuring scheme.
FinCEN's investigation revealed that the business also failed to file required Currency
Transaction Reports for customer transactions exceeding $10,000, failed to maintain required
records, and failed to implement an adequate anti-money laundering program. The business owner
was also indicted on criminal structuring charges carrying a maximum penalty of ten years
imprisonment. As part of the consent order, the business agreed to retain an independent
compliance officer and implement enhanced transaction monitoring.""",
        "metadata": {
            "fraud_types": ["Structuring"],
            "amounts_mentioned": ["$3.8 million", "$10,000", "$22 million", "$30,000"],
            "regulations": ["31 USC 5324", "31 USC 5313", "31 CFR 1020.410", "Bank Secrecy Act"],
        },
    },
    # --- Layering cases (SEC-0018 to SEC-0022) ---
    {
        "id": "SEC-0018",
        "source": "SEC Litigation Release",
        "title": "SEC Charges Three Shell Companies Used in Trade-Based Money Laundering Scheme",
        "year": "2024",
        "content": """The SEC charged three shell companies and their beneficial owners with
participating in a trade-based money laundering scheme that moved approximately $78 million
through fraudulent international trade transactions. The shell companies, registered in Delaware,
Nevada, and Wyoming, purported to be import-export businesses trading in electronics and textiles.
In reality, the companies engaged in over-invoicing and under-invoicing of goods to transfer value
across borders without moving corresponding physical goods. The scheme involved issuing invoices
for goods at five to ten times their actual value, with the excess payment representing laundered
funds. Counterparty companies in Hong Kong and the UAE issued corresponding invoices, creating
the appearance of legitimate trade. Customs records showed that actual shipments, when they
occurred, contained goods worth a fraction of the invoiced amounts. Wire transfers between the
shell companies and their foreign counterparties were routed through correspondent banking
relationships at four major U.S. banks. The beneficial owners used nominee directors and
registered agent services to conceal their identities. The SEC coordinated with FinCEN and
the FBI to freeze assets totaling $12.3 million.""",
        "metadata": {
            "fraud_types": ["Layering", "Money Laundering"],
            "amounts_mentioned": ["$78 million", "$12.3 million"],
            "regulations": ["18 USC 1956 Money Laundering", "31 USC 5318 BSA", "Section 17(a) Securities Act"],
        },
    },
    {
        "id": "SEC-0019",
        "source": "DOJ Criminal Complaint",
        "title": "Crypto Exchange Operators Indicted for Operating Unlicensed Money Transmitting Business",
        "year": "2024",
        "content": """The Department of Justice indicted two operators of a peer-to-peer
cryptocurrency exchange platform for operating an unlicensed money transmitting business and
money laundering. The platform facilitated the conversion of over $45 million in cryptocurrency
to fiat currency and vice versa without implementing any Know Your Customer or anti-money
laundering procedures. Users could exchange Bitcoin, Ethereum, and stablecoins for cash or bank
transfers with no identity verification. The operators charged a 3-5% commission on transactions
and actively marketed the platform's privacy features to attract customers seeking to avoid
detection. Law enforcement identified the platform through blockchain analysis showing that a
significant portion of incoming cryptocurrency was traceable to darknet marketplaces, ransomware
wallets, and stolen funds. The operators used a layered network of personal bank accounts, business
accounts, and cryptocurrency wallets to process customer orders. Funds received in bank accounts
were rapidly moved through multiple accounts before being withdrawn as cash or converted to
cryptocurrency. The indictment alleges that the operators processed transactions for at least
fifteen customers who were under active investigation for drug trafficking, fraud, and sanctions
evasion.""",
        "metadata": {
            "fraud_types": ["Layering", "Money Laundering"],
            "amounts_mentioned": ["$45 million"],
            "regulations": ["18 USC 1960 Unlicensed Money Transmitting", "18 USC 1956 Money Laundering"],
        },
    },
    {
        "id": "SEC-0020",
        "source": "SEC Litigation Release",
        "title": "SEC Charges Hedge Fund with Layered Transactions to Conceal Related-Party Dealings",
        "year": "2023",
        "content": """The SEC charged a hedge fund manager and two affiliated entities with using
layered transactions to conceal related-party dealings and inflate fund performance. The manager
created a chain of four intermediate entities, each ostensibly independent, to execute securities
transactions between the fund and entities the manager secretly controlled. The layering made the
transactions appear to be arm's-length dealings with unrelated counterparties. Through these
layered trades, the manager was able to sell overvalued assets to the fund at inflated prices,
generating fictitious profits of approximately $28 million reported to investors. Each intermediate
entity added a small markup to the transaction price, creating a trail of apparently legitimate
broker-to-broker trades. The entities shared office space, administrative staff, and technology
infrastructure, but were presented to auditors as independent firms. Investors were not informed
of the related-party nature of the transactions, which constituted a material breach of the fund's
operating agreements. The SEC seeks disgorgement of management fees and performance allocations
totaling $6.4 million, plus civil penalties.""",
        "metadata": {
            "fraud_types": ["Layering"],
            "amounts_mentioned": ["$28 million", "$6.4 million"],
            "regulations": ["Section 206 Investment Advisers Act", "Section 17(a) Securities Act", "Rule 10b-5"],
        },
    },
    {
        "id": "SEC-0021",
        "source": "FinCEN Enforcement Action",
        "title": "FinCEN Identifies Layering Pattern in Correspondent Banking Used for Sanctions Evasion",
        "year": "2025",
        "content": """FinCEN issued an enforcement action against a U.S. correspondent bank for
failing to detect and report a layering scheme used by a foreign financial institution to evade
OFAC sanctions. The foreign bank, domiciled in a high-risk jurisdiction, routed transactions for
sanctioned entities through a chain of four intermediary banks before reaching the U.S.
correspondent bank. By the time the transactions arrived, the originator information had been
stripped or replaced with the names of non-sanctioned entities. Over a two-year period,
approximately $320 million in transactions linked to sanctioned parties were processed through
this layered correspondent banking chain. The U.S. bank's compliance systems failed to conduct
adequate due diligence on the nested relationships within its correspondent banking portfolio.
Red flags including unusual transaction volumes, round-dollar wire transfers, and geographic
risk indicators were not investigated. FinCEN assessed a penalty of $28 million and required
the bank to terminate its relationship with the foreign financial institution. The bank was
also required to conduct a retrospective review of all transactions processed through its
correspondent banking relationships over the prior three years.""",
        "metadata": {
            "fraud_types": ["Layering", "Money Laundering"],
            "amounts_mentioned": ["$320 million", "$28 million"],
            "regulations": ["31 USC 5318 BSA", "OFAC Sanctions", "31 CFR 1010.610"],
        },
    },
    {
        "id": "SEC-0022",
        "source": "DOJ Criminal Complaint",
        "title": "Real Estate Developers Charged with Layering Scheme Using Multiple LLCs",
        "year": "2023",
        "content": """Federal prosecutors charged two real estate developers with operating a money
laundering conspiracy that used layers of limited liability companies to integrate proceeds from
a mortgage fraud scheme into the legitimate economy. The developers created over forty LLCs
registered across seven states, each with different nominee managers. Mortgage fraud proceeds
were deposited into the first layer of LLCs, then transferred through two to three additional
layers of entities before being used to purchase commercial real estate. At each layer, the funds
were commingled with legitimate business revenue to obscure their origin. The developers purchased
fourteen commercial properties valued at a total of $34 million using the layered funds. To avoid
beneficial ownership reporting requirements, no single LLC held more than one property. Title
insurance companies and closing attorneys were provided with fabricated source-of-funds
documentation. The scheme unraveled when a bank conducting enhanced due diligence on a loan
application discovered that the borrowing entity's claimed revenue could not be verified. Forensic
accounting traced the funds through the LLC network back to the original mortgage fraud proceeds.
The developers face charges of money laundering, wire fraud, and bank fraud.""",
        "metadata": {
            "fraud_types": ["Layering", "Money Laundering"],
            "amounts_mentioned": ["$34 million"],
            "regulations": ["18 USC 1956 Money Laundering", "18 USC 1343 Wire Fraud", "18 USC 1344 Bank Fraud"],
        },
    },
    # --- Identity Fraud cases (SEC-0023 to SEC-0027) ---
    {
        "id": "SEC-0023",
        "source": "DOJ Criminal Complaint",
        "title": "Synthetic Identity Fraud Ring Creates 800 Fictitious Persons for Credit Applications",
        "year": "2024",
        "content": """The Department of Justice charged nine members of a fraud ring with creating
approximately 800 synthetic identities used to fraudulently obtain credit cards, personal loans,
and lines of credit totaling over $14 million. The ring combined real Social Security numbers,
primarily belonging to children under age five and recently deceased individuals, with fabricated
names, dates of birth, and addresses. Ring members systematically built credit histories for the
synthetic identities by adding them as authorized users on existing accounts, a technique known
as piggybacking. Over twelve to eighteen months, the synthetic identities established credit
scores sufficient to obtain credit products. The ring operated a network of mail drops to receive
credit cards and correspondence. Members used the synthetic identities to open accounts at over
thirty financial institutions, diversifying exposure to avoid detection. The bust-out phase
involved maximizing credit lines through cash advances, balance transfers, and purchases of
easily liquidated goods. The ring also used synthetic identities to rent apartments and establish
utility accounts, creating additional layers of apparent legitimacy. Proceeds were laundered
through a chain of check cashing businesses controlled by the ring leader.""",
        "metadata": {
            "fraud_types": ["Identity Fraud"],
            "amounts_mentioned": ["$14 million"],
            "regulations": ["18 USC 1028 Identity Fraud", "18 USC 1029 Access Device Fraud", "18 USC 1344 Bank Fraud"],
        },
    },
    {
        "id": "SEC-0024",
        "source": "DOJ Criminal Complaint",
        "title": "Identity Theft Ring Exploits Data Breach to Open Fraudulent Bank Accounts",
        "year": "2023",
        "content": """Federal prosecutors charged six individuals with exploiting data from a major
healthcare data breach to open fraudulent bank accounts and obtain credit products. The defendants
purchased a database containing personal information of approximately 50,000 individuals from a
darknet marketplace. The stolen data included full names, Social Security numbers, dates of birth,
addresses, and employer information. Using this data, the defendants opened over 300 bank accounts
at twelve different financial institutions using the victims' real identities. To bypass identity
verification procedures, the defendants used high-quality counterfeit driver's licenses and
utility bills produced at a document fabrication facility operated by one co-conspirator. Funds
were deposited into the fraudulent accounts through mobile check deposit of counterfeit checks
and unauthorized ACH transfers from the victims' existing accounts. The defendants withdrew
funds through ATM transactions, point-of-sale purchases, and wire transfers within 48 hours of
account funding. Total losses across all financial institutions exceeded $7.8 million. Several
victims discovered the fraud only after receiving collection notices for accounts they never
opened. The FBI traced the scheme through IP addresses used to access online banking portals.""",
        "metadata": {
            "fraud_types": ["Identity Fraud", "Account Takeover"],
            "amounts_mentioned": ["$7.8 million"],
            "regulations": ["18 USC 1028 Identity Fraud", "18 USC 1030 Computer Fraud", "18 USC 1344 Bank Fraud"],
        },
    },
    {
        "id": "SEC-0025",
        "source": "FinCEN Enforcement Action",
        "title": "FinCEN Penalizes Online Lender for Inadequate Identity Verification Controls",
        "year": "2024",
        "content": """FinCEN assessed a $4.5 million civil money penalty against an online lending
platform for failing to implement adequate customer identity verification procedures, resulting
in the approval of over 2,200 loan applications submitted using stolen or synthetic identities.
The platform's automated underwriting system relied primarily on credit bureau data and basic
identity matching without implementing multi-factor authentication, document verification, or
device fingerprinting. Fraudsters exploited these weaknesses by submitting applications using
stolen personal information paired with synthetic elements. The platform disbursed approximately
$18 million in loans to fraudulent applicants over a two-year period. Recovery rates on
fraudulent loans were less than 5%. FinCEN's examination revealed that the platform received
multiple fraud reports from identity theft victims but failed to implement systemic improvements
to its verification processes. The platform's fraud detection model was not updated during the
examination period despite a fraud rate exceeding 8%, well above industry averages. Internal
communications showed that management prioritized loan volume growth over fraud prevention
measures recommended by the compliance team. The platform agreed to implement enhanced identity
verification including document authentication and biometric verification.""",
        "metadata": {
            "fraud_types": ["Identity Fraud"],
            "amounts_mentioned": ["$4.5 million", "$18 million"],
            "regulations": ["31 USC 5318 BSA", "31 CFR 1010.220 CIP", "Bank Secrecy Act"],
        },
    },
    {
        "id": "SEC-0026",
        "source": "DOJ Criminal Complaint",
        "title": "Tax Refund Fraud Scheme Using Stolen Identities of Military Personnel",
        "year": "2022",
        "content": """The DOJ charged four individuals with filing over 1,100 fraudulent federal
tax returns using stolen identities of active-duty military personnel and veterans. The defendants
obtained personal information including Social Security numbers, dates of birth, and prior-year
income data through a co-conspirator employed at a military healthcare facility. Using this
information, the defendants prepared and electronically filed fabricated tax returns claiming
refunds averaging $5,200 per return. Refunds were directed to prepaid debit cards purchased
under fictitious names. The defendants recruited associates to purchase prepaid cards and
withdraw funds from ATMs, paying them a percentage of the proceeds. Total fraudulent refunds
claimed exceeded $5.7 million, of which the IRS paid approximately $3.8 million before
detecting the pattern. The scheme was identified when multiple military personnel reported
receiving IRS notices about duplicate return filings. The IRS Criminal Investigation division
traced the prepaid card purchases to a network of retail stores and linked the purchases through
surveillance footage and transaction records. The defendants were charged with wire fraud,
identity theft, and conspiracy to defraud the United States. The lead defendant received a
sentence of 72 months in federal prison.""",
        "metadata": {
            "fraud_types": ["Identity Fraud"],
            "amounts_mentioned": ["$5,200", "$5.7 million", "$3.8 million"],
            "regulations": ["18 USC 1028 Identity Fraud", "18 USC 1343 Wire Fraud", "26 USC 7206 Tax Fraud"],
        },
    },
    {
        "id": "SEC-0027",
        "source": "DOJ Criminal Complaint",
        "title": "Business Email Compromise Ring Uses Synthetic Identities to Redirect Corporate Payments",
        "year": "2025",
        "content": """The FBI and DOJ disrupted a business email compromise ring that used
synthetic identities to open bank accounts for receiving redirected corporate payments. The ring
compromised email accounts of executives at twelve mid-size companies and intercepted
communications regarding pending vendor payments. Using spoofed emails that appeared to come
from legitimate vendors, the ring sent updated payment instructions directing funds to accounts
opened under synthetic identities at U.S. banks. The synthetic identities were created using
a combination of stolen Social Security numbers and fabricated business documentation. The ring
opened accounts under names of fictitious companies with professional-looking websites and
incorporation documents. Once payments arrived, funds were immediately transferred through
a series of domestic accounts before being wired to accounts in West Africa and Southeast Asia.
Over a nine-month period, the ring successfully redirected 34 corporate payments totaling
$11.2 million. Only $2.1 million was recovered through emergency wire recalls initiated by
victim companies. The ring maintained approximately sixty active synthetic identity bank accounts
at any given time, rotating accounts frequently to avoid detection. Enhanced due diligence by
one bank that questioned the rapid movement of funds through a newly opened account led to the
identification of the broader network.""",
        "metadata": {
            "fraud_types": ["Identity Fraud", "Wire Fraud"],
            "amounts_mentioned": ["$11.2 million", "$2.1 million"],
            "regulations": ["18 USC 1028 Identity Fraud", "18 USC 1343 Wire Fraud", "18 USC 1030 Computer Fraud"],
        },
    },
    # --- Account Takeover cases (SEC-0028 to SEC-0032) ---
    {
        "id": "SEC-0028",
        "source": "DOJ Criminal Complaint",
        "title": "Phishing Campaign Targets Online Banking Customers Leading to $3.4 Million in Losses",
        "year": "2024",
        "content": """Federal prosecutors charged seven members of a cybercrime group with operating
a large-scale phishing campaign targeting customers of major U.S. banks. The group sent over
2.5 million phishing emails designed to replicate legitimate bank communications, including
security alerts, account verification requests, and transaction notifications. The emails directed
recipients to counterfeit banking websites that captured login credentials, security questions,
and one-time passcodes. Using the harvested credentials, the group accessed approximately 1,800
victim accounts and initiated unauthorized transfers. The group employed a real-time phishing
technique in which a member would simultaneously log into the victim's actual bank account while
the victim entered credentials on the phishing site, enabling the capture of multi-factor
authentication codes. Stolen funds were transferred to money mule accounts and rapidly converted
to cryptocurrency. Total losses across all victim accounts exceeded $3.4 million. The group
operated from multiple locations and used VPN services and anonymized communication tools to
evade detection. The investigation was initiated after a bank's fraud analytics team identified
a cluster of account compromises linked to a common phishing infrastructure. Digital forensics
traced the phishing domains to a hosting provider used by the group.""",
        "metadata": {
            "fraud_types": ["Account Takeover"],
            "amounts_mentioned": ["$3.4 million"],
            "regulations": ["18 USC 1030 Computer Fraud", "18 USC 1343 Wire Fraud", "18 USC 1028A Aggravated Identity Theft"],
        },
    },
    {
        "id": "SEC-0029",
        "source": "DOJ Criminal Complaint",
        "title": "SIM Swap Fraud Ring Drains Cryptocurrency Accounts of High-Net-Worth Individuals",
        "year": "2023",
        "content": """The DOJ charged five individuals with conducting SIM swap attacks to gain
control of phone numbers belonging to high-net-worth cryptocurrency investors and drain their
exchange accounts. The ring bribed employees at two major wireless carriers to execute
unauthorized SIM transfers, redirecting victims' phone numbers to SIM cards controlled by the
defendants. With control of the phone numbers, the defendants intercepted SMS-based two-factor
authentication codes and reset passwords for cryptocurrency exchange accounts, email accounts,
and cloud storage services. Over an eight-month period, the ring targeted 43 individuals and
successfully compromised 31 accounts, stealing cryptocurrency valued at approximately $19.5
million at the time of theft. The stolen cryptocurrency was quickly moved through multiple
wallets, converted through decentralized exchanges and mixing services, and partially cashed
out through peer-to-peer transactions. The ring leader recruited participants through gaming
communities and paid the carrier employees between $500 and $2,000 per SIM swap. The scheme was
uncovered when a victim who was also a cybersecurity professional traced the SIM swap to a
specific carrier employee through analysis of account access logs and carrier records obtained
through legal process. The ring leader was sentenced to 96 months in federal prison.""",
        "metadata": {
            "fraud_types": ["Account Takeover"],
            "amounts_mentioned": ["$19.5 million", "$500", "$2,000"],
            "regulations": ["18 USC 1030 Computer Fraud", "18 USC 1343 Wire Fraud", "18 USC 1028A Aggravated Identity Theft"],
        },
    },
    {
        "id": "SEC-0030",
        "source": "SEC Litigation Release",
        "title": "Brokerage Employee Exploits Internal Access to Execute Unauthorized Trades in Client Accounts",
        "year": "2023",
        "content": """The SEC charged a registered representative at a national brokerage firm with
unauthorized trading in client accounts to generate commissions and conceal personal trading
losses. Over a fourteen-month period, the representative accessed 67 client accounts and executed
over 1,200 unauthorized trades. The representative focused on elderly clients with large account
balances and infrequent login activity. Unauthorized trades included purchases of high-commission
structured products, excessive options trading, and purchases of securities in which the
representative held personal positions to support their prices. The representative altered account
contact information to prevent clients from receiving trade confirmations and modified risk
tolerance profiles to justify the trading activity. Total unauthorized commissions generated
exceeded $890,000, and client losses from unsuitable investments exceeded $3.2 million. The
scheme was detected when a client's family member reviewing account statements noticed unfamiliar
transactions and filed a complaint. The brokerage firm's internal investigation revealed the
scope of unauthorized activity. The SEC seeks disgorgement, civil penalties, and a permanent bar
from the securities industry. The representative also faces criminal charges for wire fraud.""",
        "metadata": {
            "fraud_types": ["Account Takeover"],
            "amounts_mentioned": ["$890,000", "$3.2 million"],
            "regulations": ["Section 10(b) Exchange Act", "Rule 10b-5", "18 USC 1343 Wire Fraud"],
        },
    },
    {
        "id": "SEC-0031",
        "source": "DOJ Criminal Complaint",
        "title": "Call Center Scam Operation Takes Over Elderly Victims' Bank Accounts",
        "year": "2024",
        "content": """The DOJ and FBI charged fourteen individuals operating a fraudulent call center
with taking over bank accounts of elderly victims through social engineering. The call center
operators impersonated bank fraud department employees, IRS agents, and law enforcement officers.
Victims received phone calls claiming their accounts had been compromised and that immediate
action was required. Callers directed victims to provide account credentials, Social Security
numbers, and personal identification numbers, or to install remote access software on their
computers. Once access was obtained, operators transferred funds to accounts controlled by the
ring through wire transfers and Zelle payments. In some cases, operators convinced victims to
purchase gift cards and provide the redemption codes by phone. The operation targeted individuals
over age 65 identified through purchased marketing lists. Over a two-year period, the call center
defrauded approximately 420 victims with total losses exceeding $8.7 million. The average loss
per victim was approximately $20,700. The call center operated from leased office space and
employed approximately thirty callers working in shifts. Law enforcement identified the operation
through a pattern of victim reports and traced phone numbers to VoIP services purchased with
prepaid credit cards. Several defendants were also charged with elder fraud enhancement penalties.""",
        "metadata": {
            "fraud_types": ["Account Takeover"],
            "amounts_mentioned": ["$8.7 million", "$20,700"],
            "regulations": ["18 USC 1343 Wire Fraud", "18 USC 1028A Aggravated Identity Theft", "18 USC 2328 Elder Fraud"],
        },
    },
    {
        "id": "SEC-0032",
        "source": "DOJ Criminal Complaint",
        "title": "IT Contractor Charged with Account Takeover Using Stolen Corporate Credentials",
        "year": "2025",
        "content": """Federal prosecutors charged a former IT contractor with unauthorized access to
corporate financial systems at three companies where he had previously been engaged. The
contractor retained administrative credentials after his engagements ended and used them to access
accounts payable systems, treasury management platforms, and banking portals. Over a six-month
period, the contractor initiated unauthorized wire transfers from corporate accounts totaling
$4.1 million. The transfers were directed to accounts held by shell companies the contractor
had established using nominee identities. The contractor modified transaction logs and disabled
email notifications to delay detection of the unauthorized transfers. He also created backdoor
accounts in the companies' systems to maintain persistent access after his primary credentials
were eventually deactivated. The scheme was discovered when a controller at one victim company
noticed a discrepancy between the general ledger and bank statements during a monthly
reconciliation. Digital forensics traced the unauthorized access to IP addresses associated with
the contractor's residence and a co-working space he frequented. The contractor faces charges
of computer fraud, wire fraud, and aggravated identity theft. Approximately $1.3 million was
recovered from seized accounts.""",
        "metadata": {
            "fraud_types": ["Account Takeover"],
            "amounts_mentioned": ["$4.1 million", "$1.3 million"],
            "regulations": ["18 USC 1030 Computer Fraud", "18 USC 1343 Wire Fraud", "18 USC 1028A Aggravated Identity Theft"],
        },
    },
    # --- Insider Trading cases (SEC-0033 to SEC-0035) ---
    {
        "id": "SEC-0033",
        "source": "SEC Litigation Release",
        "title": "SEC Charges Pharmaceutical Executive and Tipping Chain of Five Traders",
        "year": "2024",
        "content": """The SEC charged a pharmaceutical company executive and five downstream traders
in a tipping chain that generated over $4.8 million in illicit profits from insider trading ahead
of an FDA drug approval announcement. The executive, who served as Vice President of Regulatory
Affairs, learned that the company's flagship drug candidate had received a favorable FDA advisory
committee recommendation three days before the public announcement. The executive tipped a close
friend, who in turn tipped three additional individuals, creating a chain of four levels of
tippees. Each participant in the chain purchased shares and call options in the pharmaceutical
company. Trading patterns showed a surge in call option purchases in the five trading days before
the announcement, with the implicated accounts representing over 40% of total call option volume
during that period. The drug approval announcement caused the stock price to increase by 62%,
generating profits ranging from $180,000 to $1.9 million for individual defendants. The SEC's
investigation was triggered by the exchange's automated surveillance system detecting abnormal
options trading activity. Phone records and encrypted messaging data established the
communication chain between the executive and downstream traders. All defendants have been
charged with violations of Section 10(b) and Rule 10b-5.""",
        "metadata": {
            "fraud_types": ["Insider Trading"],
            "amounts_mentioned": ["$4.8 million", "$180,000", "$1.9 million"],
            "regulations": ["Section 10(b) Exchange Act", "Rule 10b-5", "Section 21A Exchange Act"],
        },
    },
    {
        "id": "SEC-0034",
        "source": "SEC Litigation Release",
        "title": "Investment Banker Charged with Insider Trading in Eight M&A Transactions",
        "year": "2023",
        "content": """The SEC charged a senior investment banker at a global financial institution
with trading on material nonpublic information obtained through his role advising on mergers and
acquisitions. Over a three-year period, the banker traded in advance of eight public M&A
announcements involving companies his firm was advising, generating profits totaling $7.3 million.
The banker used a brokerage account held in the name of a college roommate domiciled overseas to
execute the trades. The roommate received 20% of the profits as compensation. To further conceal
the scheme, the banker communicated trading instructions using a prepaid phone purchased under a
false name and met the roommate in person during international business trips to exchange cash
payments. The banker's information wall crossings were documented in the firm's compliance records,
establishing his access to MNPI for each of the eight transactions. Statistical analysis
demonstrated that the probability of the trading pattern occurring by chance was less than one in
ten million. The SEC coordinated with foreign regulators to freeze the overseas brokerage account.
The banker was terminated by his employer and faces both SEC civil charges and DOJ criminal
prosecution.""",
        "metadata": {
            "fraud_types": ["Insider Trading"],
            "amounts_mentioned": ["$7.3 million"],
            "regulations": ["Section 10(b) Exchange Act", "Rule 10b-5", "Rule 10b5-1"],
        },
    },
    {
        "id": "SEC-0035",
        "source": "SEC Litigation Release",
        "title": "Corporate Accountant Trades on Advance Knowledge of Earnings Shortfall",
        "year": "2022",
        "content": """The SEC charged a senior accountant at a publicly traded retail company with
insider trading based on advance knowledge of a significant earnings shortfall. The accountant,
who was responsible for preparing quarterly financial statements, learned that the company's
fourth-quarter revenue would fall approximately 28% below analyst consensus estimates due to
weaker-than-expected holiday season sales. Two weeks before the earnings announcement, the
accountant purchased 500 put option contracts on the company's stock through an online brokerage
account. The accountant also sold short 8,000 shares of the company's stock through a separate
account. When the earnings shortfall was publicly announced, the stock price declined by 35%,
generating profits of approximately $1.1 million from the put options and $420,000 from the
short sale. The accountant's trading was detected by the company's insider trading monitoring
system, which flagged the options activity as inconsistent with the accountant's trading history.
An internal investigation confirmed that the accountant had access to preliminary financial
results prior to trading. The accountant has agreed to a settlement including disgorgement of
all profits, a civil penalty equal to the amount of profits, and a permanent officer-and-director
bar.""",
        "metadata": {
            "fraud_types": ["Insider Trading"],
            "amounts_mentioned": ["$1.1 million", "$420,000"],
            "regulations": ["Section 10(b) Exchange Act", "Rule 10b-5", "Rule 10b5-1"],
        },
    },
    # --- Money Laundering cases (SEC-0036 to SEC-0040) ---
    {
        "id": "SEC-0036",
        "source": "FinCEN Enforcement Action",
        "title": "Real Estate Money Laundering Through All-Cash Luxury Property Purchases",
        "year": "2024",
        "content": """FinCEN and the DOJ charged a network of real estate professionals and their
clients with using all-cash luxury property purchases to launder proceeds from foreign corruption
and fraud. The network facilitated the purchase of seventeen residential properties in Miami,
New York, and Los Angeles with a combined value of $92 million. All purchases were made through
anonymous LLCs, trusts, and foreign corporate entities designed to obscure the beneficial owners.
The funds used for purchases originated from government contracts obtained through bribery by
foreign public officials and from a securities fraud scheme operating in Eastern Europe. Title
companies involved in the transactions failed to conduct adequate due diligence on the source of
funds despite the all-cash nature and anonymity of the purchasers. Real estate agents facilitated
introductions to compliant title companies and provided guidance on structuring ownership to avoid
FinCEN's Geographic Targeting Orders. After purchase, several properties were renovated and resold
at market prices, completing the money laundering cycle by producing apparently legitimate real
estate sale proceeds. The investigation was conducted in coordination with law enforcement
agencies in four countries and resulted in the seizure of nine properties.""",
        "metadata": {
            "fraud_types": ["Money Laundering"],
            "amounts_mentioned": ["$92 million"],
            "regulations": ["18 USC 1956 Money Laundering", "FinCEN Geographic Targeting Order", "31 CFR 1010.380"],
        },
    },
    {
        "id": "SEC-0037",
        "source": "DOJ Criminal Complaint",
        "title": "Cryptocurrency Mixing Service Operator Charged with Money Laundering Conspiracy",
        "year": "2025",
        "content": """The Department of Justice charged the operator of a cryptocurrency mixing
service with money laundering conspiracy for processing over $600 million in cryptocurrency,
a substantial portion of which was traceable to criminal activity. The mixing service, accessible
through both the clearnet and Tor network, accepted Bitcoin and other cryptocurrencies, pooled
them with other users' funds, and returned equivalent amounts minus a fee to different wallet
addresses specified by users. The service was specifically designed to sever the blockchain trail
connecting the source and destination of funds. Law enforcement analysis determined that
approximately $180 million of the funds processed through the mixer originated from ransomware
payments, darknet marketplace transactions, and cryptocurrency theft. The operator actively
marketed the service to criminal users, maintained a presence on cybercrime forums, and offered
volume discounts for large transactions. Despite receiving multiple law enforcement inquiries and
grand jury subpoenas, the operator continued to operate the service and destroyed records in
response to legal process. The operator was arrested during international travel and extradited
to the United States. Servers hosting the mixing service were seized in a coordinated operation
with European law enforcement. The operator faces charges carrying a maximum sentence of 20 years.""",
        "metadata": {
            "fraud_types": ["Money Laundering"],
            "amounts_mentioned": ["$600 million", "$180 million"],
            "regulations": ["18 USC 1956 Money Laundering", "18 USC 1960 Unlicensed Money Transmitting"],
        },
    },
    {
        "id": "SEC-0038",
        "source": "FinCEN Enforcement Action",
        "title": "FinCEN Penalizes Correspondent Bank for Processing Suspicious Transactions from High-Risk Jurisdictions",
        "year": "2023",
        "content": """FinCEN assessed a $42 million civil money penalty against a large U.S. bank
for systemic failures in its correspondent banking anti-money laundering program. The bank
maintained correspondent relationships with over 200 foreign financial institutions, including
several in jurisdictions identified as high-risk for money laundering. Examination revealed that
the bank failed to conduct adequate due diligence on respondent banks, failed to monitor
transactions for suspicious activity, and failed to file timely Suspicious Activity Reports.
Over a five-year period, the bank processed approximately $2.8 billion in transactions through
correspondent accounts that exhibited patterns consistent with money laundering, including rapid
movement of funds through multiple accounts, transactions with no apparent business purpose, and
funds flows involving jurisdictions with weak AML controls. The bank's transaction monitoring
system was not configured to analyze correspondent banking activity at the individual transaction
level, instead relying on aggregate account-level reviews conducted only annually. Internal audit
reports had identified deficiencies in the correspondent banking program three years before the
enforcement action, but management failed to implement recommended improvements. The bank agreed
to engage an independent compliance monitor for three years and to invest $120 million in
technology and staffing improvements to its BSA compliance program.""",
        "metadata": {
            "fraud_types": ["Money Laundering"],
            "amounts_mentioned": ["$42 million", "$2.8 billion", "$120 million"],
            "regulations": ["31 USC 5318 BSA", "31 CFR 1010.610", "Bank Secrecy Act"],
        },
    },
    {
        "id": "SEC-0039",
        "source": "DOJ Criminal Complaint",
        "title": "Funnel Account Scheme Launders Drug Proceeds Through Nationwide Network",
        "year": "2024",
        "content": """The DEA and DOJ dismantled a money laundering network that used funnel
accounts to transfer drug trafficking proceeds from cities across the United States to
consolidation points near the southwest border. The network maintained over 80 bank accounts
at national and regional banks under the names of individuals recruited as account holders. Drug
proceeds collected in cities including Chicago, Atlanta, New York, and Philadelphia were deposited
in small amounts into local accounts and then transferred via electronic funds transfer to
accounts in cities near the border, including El Paso, San Diego, and Laredo. The consolidated
funds were then withdrawn as cash and physically transported across the border. The network
processed approximately $67 million over a twenty-month period. Individual funnel accounts
typically remained active for three to four months before being abandoned and replaced with new
accounts. The account holders, who were paid between $500 and $1,500 per month, were instructed
to make deposits and withdrawals that appeared consistent with normal personal banking activity.
The network was identified through FinCEN analysis of suspicious patterns across multiple banks,
including accounts in different cities controlled by related individuals showing coordinated
deposit and withdrawal activity.""",
        "metadata": {
            "fraud_types": ["Money Laundering", "Structuring"],
            "amounts_mentioned": ["$67 million", "$500", "$1,500"],
            "regulations": ["18 USC 1956 Money Laundering", "21 USC 846 Drug Conspiracy", "31 USC 5324"],
        },
    },
    {
        "id": "SEC-0040",
        "source": "FinCEN Enforcement Action",
        "title": "Casino Penalized for Failing to Report Suspicious Gambling Activity Linked to Money Laundering",
        "year": "2023",
        "content": """FinCEN assessed a $8.5 million civil money penalty against a Las Vegas casino
resort for failing to implement adequate anti-money laundering controls and failing to report
suspicious gambling activity. The casino failed to file Suspicious Activity Reports on numerous
high-value patrons who displayed indicators of money laundering, including patrons who conducted
minimal gambling relative to the volume of cash transactions, patrons who purchased chips with
cash and redeemed them for checks without significant play, and patrons who conducted cash-in and
cash-out transactions on the same visit with minimal net gambling activity. Over a three-year
examination period, the casino processed over $1.2 billion in cash transactions but filed only
a fraction of the SARs required by the volume and nature of its cash activity. The casino's
patron due diligence program failed to identify beneficial owners of entities holding casino
credit lines. Foreign nationals from high-risk jurisdictions were extended credit lines exceeding
$500,000 without adequate source-of-funds verification. Several patrons later identified in
criminal investigations had conducted millions of dollars in transactions at the casino without
triggering any suspicious activity reporting. The casino agreed to comprehensive remedial measures
including enhanced patron due diligence and upgraded transaction monitoring systems.""",
        "metadata": {
            "fraud_types": ["Money Laundering"],
            "amounts_mentioned": ["$8.5 million", "$1.2 billion", "$500,000"],
            "regulations": ["31 USC 5318 BSA", "31 CFR 1021.320", "Bank Secrecy Act"],
        },
    },
    # --- Legitimate / False Alarm cases (SEC-0041 to SEC-0045) ---
    {
        "id": "SEC-0041",
        "source": "SEC Litigation Release",
        "title": "Seasonal Business Deposit Pattern Cleared After Investigation of Structuring Alert",
        "year": "2024",
        "content": """A transaction monitoring system generated a structuring alert for a landscaping
and snow removal company that made frequent cash deposits in amounts ranging from $6,000 to
$9,800 during the spring and summer months. The alert flagged 47 deposits over a four-month
period with an aggregate value of $378,000. Investigation determined that the deposit pattern
was consistent with the company's legitimate business operations. The company serviced
approximately 200 residential customers who paid in cash or check for weekly lawn maintenance.
Revenue was deposited two to three times per week based on collection schedules. The deposit
amounts varied naturally based on customer payment timing and service frequency. During winter
months, the company's deposit pattern shifted to reflect snow removal revenue, which was collected
differently. The company provided three years of tax returns showing consistent reported revenue,
customer lists with corresponding service agreements, and employee payroll records. The bank's
investigation confirmed that deposit amounts were proportional to the company's documented
customer base and service pricing. The alert was closed with no SAR filing, and the account
was reclassified with an updated customer risk profile reflecting the cash-intensive nature of
the business.""",
        "metadata": {
            "fraud_types": ["Legitimate"],
            "amounts_mentioned": ["$6,000", "$9,800", "$378,000"],
            "regulations": [],
        },
    },
    {
        "id": "SEC-0042",
        "source": "SEC Litigation Release",
        "title": "International Wire Transfers for Tuition Payments Cleared After Enhanced Review",
        "year": "2023",
        "content": """A bank's compliance department initiated an enhanced review of a customer
account that received a series of large international wire transfers totaling $245,000 over a
six-month period. The transfers originated from three different countries: China, South Korea,
and India. Each transfer ranged from $25,000 to $55,000 and was sent by different originators.
The account holder, a U.S. resident, had no prior history of receiving international transfers
of this magnitude. Initial analysis flagged the transactions due to multiple foreign originators,
high aggregate value, deviation from historical patterns, and geographic diversity of origins.
Investigation revealed that the account holder operated a licensed educational consulting business
that assisted international students with university admissions and housing placement. The wire
transfers were tuition and living expense payments from students' families, consistent with the
company's business model. The account holder provided contracts with each student family,
university enrollment confirmations for the students, and records of corresponding disbursements
to universities and landlords. The business had been operating for three years with proper
licensing and tax filings. The review was closed with no adverse findings.""",
        "metadata": {
            "fraud_types": ["Legitimate"],
            "amounts_mentioned": ["$245,000", "$25,000", "$55,000"],
            "regulations": [],
        },
    },
    {
        "id": "SEC-0043",
        "source": "SEC Litigation Release",
        "title": "Cryptocurrency Trading Activity Cleared After Compliance Investigation",
        "year": "2024",
        "content": """A compliance investigation was triggered when a personal checking account
exhibited a pattern of high-frequency, high-value transfers between the account and multiple
cryptocurrency exchange accounts. Over a three-month period, the account holder made 89 transfers
totaling $1.2 million to accounts at four different cryptocurrency exchanges, and received
92 transfers totaling $1.35 million back from the same exchanges. The activity was flagged due
to the volume of transfers, involvement of multiple cryptocurrency platforms, and the rapid
movement of funds. Investigation revealed that the account holder was a professional day trader
specializing in cryptocurrency arbitrage. The trader exploited price differences between
exchanges by purchasing cryptocurrency on one exchange and selling on another. Supporting
documentation included trading records from each exchange showing corresponding buy and sell
transactions, a Schedule C filed with the IRS reporting cryptocurrency trading as a business,
profit and loss statements prepared by the trader's accountant, and records from the prior two
tax years showing similar trading activity. The net profit from trading during the flagged period
was approximately $150,000, which was fully documented and consistent with the trader's reported
income. No suspicious activity was identified.""",
        "metadata": {
            "fraud_types": ["Legitimate"],
            "amounts_mentioned": ["$1.2 million", "$1.35 million", "$150,000"],
            "regulations": [],
        },
    },
    {
        "id": "SEC-0044",
        "source": "SEC Litigation Release",
        "title": "Estate Distribution Transfers Cleared After Alert for Unusual Account Activity",
        "year": "2023",
        "content": """A bank's fraud detection system flagged a series of outgoing wire transfers
from a newly opened trust account. Within two weeks of account opening, seven wire transfers
totaling $1.8 million were sent to individual accounts at different banks. The transfers ranged
from $120,000 to $450,000. The activity was flagged due to the new account status, rapid outflows,
multiple beneficiaries, and large aggregate amount. Investigation determined the transfers were
legitimate estate distributions. The account was a revocable trust account opened by the executor
of a recently deceased individual's estate. The wire transfers were distributions to seven
beneficiaries named in the decedent's will. The executor provided the death certificate, letters
testamentary issued by the probate court, a copy of the will identifying all beneficiaries and
their respective shares, and closing documents from the sale of the decedent's residence which
funded the trust account. The distribution amounts exactly matched the proportional shares
specified in the will after deducting estate expenses and taxes. The executor was a licensed
attorney with a verified law practice. The investigation was closed with no further action.""",
        "metadata": {
            "fraud_types": ["Legitimate"],
            "amounts_mentioned": ["$1.8 million", "$120,000", "$450,000"],
            "regulations": [],
        },
    },
    {
        "id": "SEC-0045",
        "source": "SEC Litigation Release",
        "title": "Medical Practice Deposit Pattern Investigated and Cleared",
        "year": "2022",
        "content": """An automated monitoring system flagged a medical practice's business account
for making daily cash deposits that appeared to be structured below reporting thresholds. Over a
two-month period, the practice made 42 cash deposits averaging $8,900 each, for a total of
approximately $374,000. No single deposit exceeded $10,000. The practice was an urgent care
clinic in a predominantly unbanked community where a significant portion of patients paid for
services in cash. The practice provided records showing that cash payments represented
approximately 40% of total revenue, consistent with the demographics of the patient population.
Daily cash collections varied between $5,000 and $12,000 depending on patient volume. Deposits
were made at the end of each business day by the office manager as part of routine cash handling
procedures. On days when cash collections exceeded $10,000, the full amount was deposited and
CTRs were properly filed by the bank. The practice provided three years of consistent deposit
history, tax returns, patient visit records showing cash versus insurance billing, and
documentation of its cash handling procedures. The investigation concluded that the deposit
pattern reflected legitimate business operations rather than structuring activity.""",
        "metadata": {
            "fraud_types": ["Legitimate"],
            "amounts_mentioned": ["$8,900", "$374,000", "$10,000", "$5,000", "$12,000"],
            "regulations": [],
        },
    },
    # --- Wire Fraud cases (SEC-0046 to SEC-0048) ---
    {
        "id": "SEC-0046",
        "source": "DOJ Criminal Complaint",
        "title": "Business Email Compromise Scheme Targets Corporate Treasury Departments",
        "year": "2024",
        "content": """The FBI charged twelve individuals in connection with a business email
compromise scheme that defrauded 28 companies of approximately $23 million through fraudulent
wire transfer requests. The conspirators used spearphishing emails to gain access to the email
accounts of senior executives, including CEOs and CFOs, at targeted companies. Once inside the
email systems, the conspirators monitored communications for weeks to understand payment processes,
vendor relationships, and communication styles. They then sent fraudulent wire transfer
instructions from the compromised accounts to treasury department employees, requesting urgent
transfers to accounts controlled by the conspiracy. The emails mimicked the writing style and
formatting of the compromised executives and referenced actual pending transactions to increase
credibility. Receiving accounts were opened at U.S. banks using fraudulent identification
documents and were drained within hours of receiving the wire transfers. The conspirators also
intercepted legitimate vendor invoices and modified banking details before forwarding them to the
victim companies. The average fraud loss per victim company was $821,000. Only $4.2 million was
recovered through wire recall requests. The investigation revealed that the scheme operated from
both domestic and international locations, with money mules in the U.S. and coordinators overseas.""",
        "metadata": {
            "fraud_types": ["Wire Fraud"],
            "amounts_mentioned": ["$23 million", "$821,000", "$4.2 million"],
            "regulations": ["18 USC 1343 Wire Fraud", "18 USC 1349 Conspiracy", "18 USC 1028A Aggravated Identity Theft"],
        },
    },
    {
        "id": "SEC-0047",
        "source": "DOJ Criminal Complaint",
        "title": "Attorney Charged with Wire Fraud for Diverting Client Settlement Funds",
        "year": "2023",
        "content": """Federal prosecutors charged a personal injury attorney with wire fraud for
diverting client settlement funds to personal accounts and gambling operations. Over a five-year
period, the attorney received settlement payments totaling $16.4 million on behalf of 73 clients
into his firm's trust account. Rather than distributing the funds to clients after deducting
legitimate fees and expenses, the attorney transferred approximately $5.8 million to personal
bank accounts and online gambling platforms. The attorney concealed the misappropriation by
providing clients with fabricated settlement statements showing lower recovery amounts than
actually received and by delaying distributions with false claims about processing requirements.
When clients demanded payment, the attorney used funds from newer settlements to pay older
obligations in a Ponzi-like fashion. The scheme collapsed when multiple clients simultaneously
demanded their settlement proceeds and the trust account had insufficient funds. A state bar
investigation triggered by client complaints uncovered the full scope of the misappropriation.
Wire fraud charges were filed based on the use of interstate wire transfers to move misappropriated
funds. The attorney's law license was immediately suspended, and he faces up to twenty years in
prison on each wire fraud count. Affected clients filed claims with the state's client protection
fund.""",
        "metadata": {
            "fraud_types": ["Wire Fraud"],
            "amounts_mentioned": ["$16.4 million", "$5.8 million"],
            "regulations": ["18 USC 1343 Wire Fraud", "18 USC 1349 Conspiracy"],
        },
    },
    {
        "id": "SEC-0048",
        "source": "DOJ Criminal Complaint",
        "title": "Romance Scam Network Defrauds Victims Through Wire Transfers and Gift Cards",
        "year": "2025",
        "content": """The DOJ and Postal Inspection Service charged eight individuals with operating
a romance scam network that defrauded over 200 victims of approximately $12.5 million through
wire transfers, gift card purchases, and cryptocurrency payments. The network created fictitious
profiles on dating websites and social media platforms, using stolen photographs of attractive
individuals. Conspirators cultivated online relationships with victims over weeks to months,
eventually fabricating emergencies requiring financial assistance, including medical crises,
business failures, customs fees for international travel, and legal problems. Victims were
instructed to send money through wire transfers to accounts controlled by the network, purchase
gift cards and provide redemption codes, or send cryptocurrency to designated wallet addresses.
Some victims were persuaded to take out personal loans or drain retirement accounts. The average
loss per victim was approximately $62,500, with individual losses ranging from $5,000 to $680,000.
Wire transfers were sent to funnel accounts at U.S. banks, where funds were rapidly withdrawn or
forwarded to accounts overseas. Gift card proceeds were converted to cash through resale
operations. The network operated from both domestic locations and West Africa, with U.S.-based
members serving primarily as money mules and account managers. Victims ranged in age from 35 to
82, with the majority being between 55 and 70 years old.""",
        "metadata": {
            "fraud_types": ["Wire Fraud"],
            "amounts_mentioned": ["$12.5 million", "$62,500", "$5,000", "$680,000"],
            "regulations": ["18 USC 1343 Wire Fraud", "18 USC 1349 Conspiracy", "18 USC 1028A Aggravated Identity Theft"],
        },
    },
    # --- Investment Fraud cases (SEC-0049 to SEC-0051) ---
    {
        "id": "SEC-0049",
        "source": "SEC Litigation Release",
        "title": "SEC Charges Operator of $150 Million Ponzi Scheme Targeting Retirement Investors",
        "year": "2024",
        "content": """The SEC charged the operator of an investment fund with operating a Ponzi
scheme that raised approximately $150 million from over 600 investors, primarily retirees seeking
stable income. The fund operator promised guaranteed annual returns of 8-12% from a purported
portfolio of commercial real estate loans and corporate bonds. Marketing materials emphasized the
safety of the investment and the operator's 25 years of financial industry experience. In reality,
the fund generated minimal legitimate investment returns. The operator used new investor capital
to pay purported returns to existing investors and to fund personal expenditures including a $4.5
million waterfront home, luxury vehicles, private jet travel, and a yacht. Investor statements
were fabricated to show consistent positive returns and a growing portfolio value. When redemption
requests increased during an economic downturn, the operator imposed withdrawal restrictions,
claiming liquidity constraints in the underlying assets. An SEC examination triggered by investor
complaints revealed that the fund held less than $8 million in actual investments against $150
million in reported assets. The operator attempted to destroy records and flee the country but was
apprehended at an international airport. The SEC obtained an emergency asset freeze and appointed
a receiver to marshal remaining assets for distribution to victims.""",
        "metadata": {
            "fraud_types": ["Ponzi Scheme", "Investment Fraud"],
            "amounts_mentioned": ["$150 million", "$4.5 million", "$8 million"],
            "regulations": ["Section 17(a) Securities Act", "Section 10(b) Exchange Act", "Rule 10b-5", "Section 206 Investment Advisers Act"],
        },
    },
    {
        "id": "SEC-0050",
        "source": "SEC Litigation Release",
        "title": "Affinity Fraud Scheme Targets Religious Community with Fake Cryptocurrency Investment",
        "year": "2023",
        "content": """The SEC charged two individuals with orchestrating an affinity fraud scheme
that raised over $35 million from members of a religious community through a fraudulent
cryptocurrency investment program. The promoters, who were active members of the community,
marketed the investment through church gatherings, community events, and word-of-mouth referrals.
They claimed to have developed a proprietary cryptocurrency trading algorithm that generated
consistent daily returns of 1-2%. Investors were told their funds would be managed by a team
of expert traders using artificial intelligence and machine learning. Early investors received
returns funded by new investor capital, reinforcing the scheme's credibility within the community.
The promoters leveraged the trust and social bonds within the religious community to discourage
due diligence and suppress skepticism. Investors who raised questions were told that doubting the
investment opportunity demonstrated a lack of faith in the community. Over $25 million of the
funds raised were diverted to the promoters' personal use, including luxury real estate and
investments in unrelated businesses. The remaining funds were lost through actual cryptocurrency
trading by inexperienced staff. The scheme collapsed when the promoters could no longer sustain
payments and abruptly ceased communications. Approximately 450 community members lost their
investments, with many losing their life savings.""",
        "metadata": {
            "fraud_types": ["Investment Fraud", "Ponzi Scheme"],
            "amounts_mentioned": ["$35 million", "$25 million"],
            "regulations": ["Section 17(a) Securities Act", "Section 5 Securities Act", "Section 10(b) Exchange Act"],
        },
    },
    {
        "id": "SEC-0051",
        "source": "SEC Litigation Release",
        "title": "SEC Halts Fraudulent Private Placement Offering Promising Oil and Gas Returns",
        "year": "2022",
        "content": """The SEC obtained an emergency restraining order halting a fraudulent private
placement offering that raised $28 million from 180 investors through false promises of returns
from oil and gas drilling operations. The promoter, a former oil industry executive, offered
limited partnership interests promising 300% returns within three years from oil production in
West Texas leases. The offering memorandum contained fabricated geological surveys, inflated
production estimates, and fictitious endorsements from energy industry experts. In reality, the
leases held by the partnership had been evaluated by independent geologists as having minimal
commercial potential. Only $4 million of investor funds were actually used for drilling
operations, which produced negligible output. The remaining $24 million was used to pay sales
commissions of 15-20% to an unlicensed sales force, fund the promoter's personal expenses, and
make Ponzi-like distributions to early investors. The promoter had previously been subject to
state securities enforcement actions in two states, a fact not disclosed to investors. The
offering was not registered with the SEC and did not qualify for any registration exemption.
Investor funds were held in accounts at two banks that failed to detect the commingling of investor
funds with the promoter's personal accounts. The SEC seeks full disgorgement, penalties, and
permanent injunctions.""",
        "metadata": {
            "fraud_types": ["Investment Fraud"],
            "amounts_mentioned": ["$28 million", "$4 million", "$24 million"],
            "regulations": ["Section 5 Securities Act", "Section 17(a) Securities Act", "Section 10(b) Exchange Act", "Rule 10b-5"],
        },
    },
    # --- Additional diverse cases (SEC-0052 to SEC-0055) ---
    {
        "id": "SEC-0052",
        "source": "DOJ Criminal Complaint",
        "title": "PPP Loan Fraud Ring Submits Fraudulent Applications Using Stolen Business Identities",
        "year": "2022",
        "content": """The DOJ charged eight individuals with submitting over 120 fraudulent
Paycheck Protection Program loan applications using stolen business identities, resulting in
the disbursement of approximately $9.2 million in federal relief funds. The defendants obtained
Employer Identification Numbers and business registration information for dormant and dissolved
companies from public databases. Using this information, they prepared fraudulent loan
applications that overstated payroll expenses using fabricated IRS tax forms and payroll records.
Applications were submitted through multiple SBA-approved lenders to reduce the likelihood of
detection. Approved funds were deposited into bank accounts opened under the stolen business
identities with fraudulent corporate documentation. Within days of receiving the funds, the
defendants transferred the money through a series of personal accounts, purchased luxury goods
including vehicles and jewelry, and withdrew large amounts as cash. Several defendants submitted
multiple applications using different stolen business identities, sometimes receiving two or three
PPP loans within the same month. The fraud was detected through SBA's post-disbursement review
process, which identified multiple loans associated with the same IP addresses and bank accounts.
The defendants face charges of wire fraud, bank fraud, and aggravated identity theft, each
carrying significant federal prison sentences.""",
        "metadata": {
            "fraud_types": ["Wire Fraud", "Identity Fraud"],
            "amounts_mentioned": ["$9.2 million"],
            "regulations": ["18 USC 1343 Wire Fraud", "18 USC 1344 Bank Fraud", "18 USC 1028 Identity Fraud"],
        },
    },
    {
        "id": "SEC-0053",
        "source": "SEC Litigation Release",
        "title": "SEC Charges Crypto Token Issuer with Unregistered Securities Offering and Fraud",
        "year": "2024",
        "content": """The SEC charged the founders of a blockchain technology company with conducting
an unregistered securities offering and fraud in connection with the sale of digital tokens that
raised $42 million from thousands of investors worldwide. The founders marketed the tokens through
social media, online forums, and a professional website as an investment in a revolutionary
decentralized finance protocol that would generate passive income through automated yield farming.
The white paper described sophisticated smart contract technology and partnerships with major
financial institutions, none of which existed. Celebrity endorsements featured on the website were
fabricated without the knowledge or consent of the individuals depicted. Of the $42 million
raised, approximately $30 million was diverted to the founders' personal accounts and used to
purchase real estate, luxury goods, and other cryptocurrencies. The remaining funds were used for
marketing to attract additional investors. No meaningful technology development occurred. The token
price collapsed by 95% when the founders became unreachable and the project website went offline.
The SEC determined that the tokens constituted investment contracts under the Howey test and that
the offering violated registration requirements. The founders were located through blockchain
analysis and international cooperation with law enforcement.""",
        "metadata": {
            "fraud_types": ["Investment Fraud"],
            "amounts_mentioned": ["$42 million", "$30 million"],
            "regulations": ["Section 5 Securities Act", "Section 17(a) Securities Act", "Section 10(b) Exchange Act"],
        },
    },
    {
        "id": "SEC-0054",
        "source": "FinCEN Enforcement Action",
        "title": "Money Services Business Penalized for Processing Transactions for Unlicensed Operators",
        "year": "2023",
        "content": """FinCEN assessed a $6.2 million civil money penalty against a registered money
services business for knowingly processing transactions on behalf of unlicensed money
transmitters. The MSB operated a network of retail agents providing wire transfer and check
cashing services. Examination revealed that several agents were processing transactions for
third-party operators who were not registered with FinCEN and did not hold required state
licenses. These unlicensed operators collected cash from customers, transported it to the MSB's
agents, and directed wire transfers to recipients in Central America and West Africa. The
unlicensed operators charged their customers fees of 8-15%, well above the MSB's standard fee
schedule, and pocketed the difference. Transaction volumes through the implicated agents were
three to five times higher than comparable agents in similar locations. The MSB's compliance
department received multiple reports from agents about the third-party operators but failed to
investigate or terminate the relationships. Over a two-year period, the unlicensed operators
processed approximately $48 million through the MSB's network. FinCEN determined that many of
the transactions were consistent with remittance-based money laundering, with funds originating
from cash-intensive businesses with indicators of unreported income. The MSB agreed to terminate
all relationships with implicated agents and implement enhanced agent oversight procedures.""",
        "metadata": {
            "fraud_types": ["Money Laundering"],
            "amounts_mentioned": ["$6.2 million", "$48 million"],
            "regulations": ["31 USC 5330", "18 USC 1960 Unlicensed Money Transmitting", "Bank Secrecy Act"],
        },
    },
    {
        "id": "SEC-0055",
        "source": "SEC Litigation Release",
        "title": "Legitimate Venture Capital Distribution Cleared After Wire Transfer Alert",
        "year": "2024",
        "content": """A bank's transaction monitoring system generated an alert for a series of
large wire transfers from a business account that had been relatively dormant for two years. Over
a three-week period, the account disbursed $6.4 million through eleven wire transfers to
individual accounts at various banks. Transfer amounts ranged from $180,000 to $1.2 million. The
account had maintained a steady balance of approximately $50,000 during the dormant period before
receiving a single incoming wire of $6.5 million that triggered the subsequent outgoing activity.
Investigation revealed that the account belonged to a venture capital fund that had recently
completed the sale of a portfolio company. The incoming $6.5 million wire represented the fund's
share of the sale proceeds distributed by an escrow agent. The outgoing transfers were returns of
capital and profit distributions to the fund's eleven limited partners, consistent with the fund's
partnership agreement. The fund manager provided the executed purchase agreement for the portfolio
company sale, escrow closing statements, the fund's limited partnership agreement specifying
distribution waterfalls, K-1 tax forms prepared for each limited partner, and correspondence from
the fund's legal counsel regarding the distribution. All distribution amounts matched the
contractual terms. The account's prior dormancy was explained by the fund being in its harvest
period with no new investments. The investigation was closed with no adverse findings.""",
        "metadata": {
            "fraud_types": ["Legitimate"],
            "amounts_mentioned": ["$6.4 million", "$180,000", "$1.2 million", "$6.5 million", "$50,000"],
            "regulations": [],
        },
    },
]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for case in SAMPLE_CASES:
        case["content_length"] = len(case["content"])
        filepath = DATA_DIR / f"sec_case_{case['id'].split('-')[1]}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(case, f, ensure_ascii=False, indent=2)

    # 요약
    summary = {
        "total_cases": len(SAMPLE_CASES),
        "cases_with_content": len(SAMPLE_CASES),
        "fraud_type_distribution": {},
        "source": "Structured from real SEC/FinCEN enforcement patterns",
    }
    for case in SAMPLE_CASES:
        for ft in case["metadata"]["fraud_types"]:
            summary["fraud_type_distribution"][ft] = (
                summary["fraud_type_distribution"].get(ft, 0) + 1
            )

    with open(DATA_DIR / "crawl_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated {len(SAMPLE_CASES)} fraud cases")
    print(f"Fraud types: {summary['fraud_type_distribution']}")
    print(f"Saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
