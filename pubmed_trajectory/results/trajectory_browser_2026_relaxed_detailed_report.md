# Detailed Trajectory Data Quality Report

## Overall

- Total records: 23195
- Records with prior guidelines: 2920 (12.6%)
- Records without prior guidelines: 20275 (87.4%)
- Duplicate topic groups: 1664
- Duplicate title groups: 551
- Cross-category PMID collisions: 114
- Cross-category title collisions: 166
- Reused prior titles across records: 427

## By Corpus

- Commercial: 13002 records; 1134 with priors (8.7%); 11868 without priors; 8921 missing current organizations; 128 invalid current years; median prior depth 0.0; max prior depth 11
- Non-commercial: 8790 records; 1586 with priors (18.0%); 7204 without priors; 3711 missing current organizations; 157 invalid current years; median prior depth 0.0; max prior depth 20
- Other: 1403 records; 200 with priors (14.3%); 1203 without priors; 627 missing current organizations; 17 invalid current years; median prior depth 0; max prior depth 11

## Problem Inventory

- no prior guidelines: 20275
- missing current organization: 13259
- missing prior pmid: 2333
- missing or placeholder prior title: 562
- missing or placeholder topic: 500
- invalid current year: 302
- missing or placeholder title: 140
- same year prior: 84
- duplicate prior title within record: 60
- missing prior organization: 20
- future prior: 3

## Incompleteness: no prior-guideline chain

- Count: 20275
- Representative examples:
- `comm` PMC522748 (2004): Allermatch™, a webtool for the prediction of potential allergenicity according to current FAO/WHO Codex alimentarius guidelines
- `comm` PMC522817 (2004): Neoadjuvant or adjuvant therapy for resectable esophageal cancer: a clinical practice guideline
- `comm` PMC526186 (2004): Radiotherapy fractionation for the palliation of uncomplicated painful bone metastases – an evidence-based practice guideline
- `comm` PMC535864 (2004): Family physicians' perspectives on practice guidelines related to cancer control
- `comm` PMC535932 (2004): Cost-effectiveness of an intensive group training protocol compared to physiotherapy guideline care for sub-acute and chronic low back pain: design of a randomised controlled trial with an economic evaluation. [ISRCTN45641649]
- `comm` PMC544887 (2005): Sicily statement on evidence-based practice
- `comm` PMC553981 (0): Unknown
- `comm` PMC555572 (2005): Reliability and validity of the AGREE instrument used by physical therapists in assessment of clinical practice guidelines

## Incompleteness: current organization missing

- Count: 13259
- Representative examples:
- `comm` PMC522748 (2004): Allermatch™, a webtool for the prediction of potential allergenicity according to current FAO/WHO Codex alimentarius guidelines
- `comm` PMC535864 (2004): Family physicians' perspectives on practice guidelines related to cancer control
- `comm` PMC535932 (2004): Cost-effectiveness of an intensive group training protocol compared to physiotherapy guideline care for sub-acute and chronic low back pain: design of a randomised controlled trial with an economic evaluation. [ISRCTN45641649]
- `comm` PMC544887 (2005): Sicily statement on evidence-based practice
- `comm` PMC553981 (0): Unknown
- `comm` PMC555572 (2005): Reliability and validity of the AGREE instrument used by physical therapists in assessment of clinical practice guidelines
- `comm` PMC1087203 (2005): Trends in the Prescribing of Thiazides for Hypertension
- `comm` PMC1087212 (2005): First-Line First? Trends in Thiazide Prescribing for Hypertensive Seniors

## Incompleteness: prior guideline missing PMID

- Count: 2333
- Representative examples:
- `comm` PMC1090562 (2005): Clinical practice guideline on the optimal radiotherapeutic management of brain metastases -> prior `Practice guideline on the management of single brain metastases` year=2002 pmid=0 org=Cancer Care Ontario Neuro-Oncology Disease Site Group (NDSG)
- `comm` PMC1479370 (2006): Indications for conservative management of scoliosis (guidelines) -> prior `Indications for conservative management of scoliosis (guidelines) [first version on SOSORT homepage]` year=2005 pmid=0 org=Society on Scoliosis Orthopaedic and Rehabilitation Treatment (SOSORT)
- `comm` PMC2151254 (1998): United Kingdom Co-ordinating Committee on Cancer Research (UKCCCR) Guidelines for the Welfare of Animals in Experimental Neoplasia (Second Edition). -> prior `Unknown` year=0 pmid=0 org=United Kingdom Co-ordinating Committee on Cancer Research (UKCCCR)
- `comm` PMC2644625 (2009): A framework for the organization and delivery of systemic treatment -> prior `Regional Models of Care for Systemic Treatment: Standards for the Organization and Delivery of Systemic Treatment` year=2007 pmid=0 org=Cancer Care Ontario
- `comm` PMC2644625 (2009): A framework for the organization and delivery of systemic treatment -> prior `Systemic Therapy Task Force Report` year=2000 pmid=0 org=Cancer Care Ontario
- `comm` PMC2701558 (2007): Guideline for Management of Postmeal Glucose -> prior `Global Guideline for Type 2 Diabetes` year=2005 pmid=0 org=International Diabetes Federation
- `comm` PMC2753009 (2003): Summary version of the standards, options and recommendations for nonmetastatic breast cancer (updated January 2001) -> prior `Fédération Nationale des Centres de Lutte Contre le Cancer, ed` year=2001 pmid=0 org=Fédération Nationale des Centres de Lutte Contre le Cancer (FNCLCC)
- `comm` PMC2753009 (2003): Summary version of the standards, options and recommendations for nonmetastatic breast cancer (updated January 2001) -> prior `Fédération Nationale des Centres de Lutte Center le Cancer, ed` year=1996 pmid=0 org=Fédération Nationale des Centres de Lutte Contre le Cancer (FNCLCC)

## Incompleteness: prior guideline title is missing or placeholder

- Count: 562
- Representative examples:
- `comm` PMC2151254 (1998): United Kingdom Co-ordinating Committee on Cancer Research (UKCCCR) Guidelines for the Welfare of Animals in Experimental Neoplasia (Second Edition). -> prior `Unknown` year=0 pmid=0 org=United Kingdom Co-ordinating Committee on Cancer Research (UKCCCR)
- `comm` PMC2773534 (2008): Infection Control in Anaesthesia -> prior `Unknown` year=2002 pmid=0 org=Association of Anaesthetists of Great Britain and Ireland
- `comm` PMC3082210 (2009): Suspected Anaphylactic Reactions Associated with Anaesthesia -> prior `Unknown` year=1990 pmid=0 org=Association of Anaesthetists of Great Britain and Ireland
- `comm` PMC3082210 (2009): Suspected Anaphylactic Reactions Associated with Anaesthesia -> prior `Unknown` year=1995 pmid=0 org=Association of Anaesthetists of Great Britain and Ireland (published jointly with the British Society for Allergy and Clinical Immunology)
- `comm` PMC3082210 (2009): Suspected Anaphylactic Reactions Associated with Anaesthesia -> prior `Unknown` year=2003 pmid=0 org=Association of Anaesthetists of Great Britain and Ireland (published jointly with the British Society for Allergy and Clinical Immunology)
- `comm` PMC3194537 (2010): German evidence and consensus based guidelines 2010 for the treatment of juvenile idiopathic arthritis (JIA) -> prior `Unknown` year=2005 pmid=0 org=Arbeitsgemeinschaft der Wissenschaftlichen Medizinischen Fachgesellschaften (AWMF)
- `comm` PMC3194537 (2010): German evidence and consensus based guidelines 2010 for the treatment of juvenile idiopathic arthritis (JIA) -> prior `Unknown` year=2008 pmid=0 org=Arbeitsgemeinschaft der Wissenschaftlichen Medizinischen Fachgesellschaften (AWMF)
- `comm` PMC3266527 (2012): Reference programme: Diagnosis and treatment of headache disorders and facial pain. Danish Headache Society, 2nd Edition, 2012 -> prior `Unknown` year=1994 pmid=0 org=Danish Headache Society

## Inconsistency: current year is invalid

- Count: 302
- Representative examples:
- `comm` PMC553981 (0): Unknown
- `comm` PMC2448601 (0): Unknown
- `comm` PMC2721507 (0): EAU guidelines for management of penile cancer
- `comm` PMC2827473 (0): Guidelines; from foe to friend? Comparative interviews with GPs in Norway and Denmark
- `comm` PMC2893080 (0): How to translate therapeutic recommendations in clinical practice guidelines into rules for critiquing physician prescriptions? Methods and application to five guidelines
- `comm` PMC3105026 (0): Unknown
- `comm` PMC3461436 (0): Parent attitudes, family dynamics and adolescent drinking: qualitative study of the Australian parenting guidelines for adolescent alcohol use
- `comm` PMC3563952 (0): Unknown

## Chronology issue: prior guideline has the same year as current guidance

- Count: 84
- Representative examples:
- `comm` PMC3936126 (2013): Guidelines for laparoscopic treatment of ventral and incisional abdominal wall hernias (International Endohernia Society [IEHS])—Part III -> prior `Guidelines for laparoscopic treatment of ventral and incisional abdominal wall hernias (International Endohernia Society [IEHS])—Part I` year=2013 pmid=0 org=International Endohernia Society
- `comm` PMC3936126 (2013): Guidelines for laparoscopic treatment of ventral and incisional abdominal wall hernias (International Endohernia Society [IEHS])—Part III -> prior `Guidelines for laparoscopic treatment of ventral and incisional abdominal wall hernias (International Endohernia Society [IEHS])—Part II` year=2013 pmid=0 org=International Endohernia Society
- `comm` PMC7184243 (2020): Society for Cardiovascular Magnetic Resonance (SCMR) guidance for the practice of cardiovascular magnetic resonance during the COVID-19 pandemic -> prior `SCMR’s COVID-19 Preparedness Toolkit` year=2020 pmid=0 org=Society for Cardiovascular Magnetic Resonance
- `comm` PMC7184547 (2020): Recommendations from the CSO-HNS taskforce on performance of tracheotomy during the COVID-19 pandemic -> prior `Guidance for Health Care Workers Performing Aerosol Generating Medical Procedures during the COVID-19 Pandemic` year=2020 pmid=0 org=Canadian Society of Otolaryngology - Head & Neck Surgery (CSO-HNS)
- `comm` PMC7429418 (2020): Preparation for the next COVID-19 wave: The European Hip Society and European Knee Associates recommendations -> prior `Resuming elective hip and knee replacement in the setting of the COVID-19 pandemic: The European Hip Society and European Knee Associates recommendations` year=2020 pmid=0 org=European Hip Society and European Knee Associates
- `comm` PMC7472403 (2020): Chemoprophylaxis, diagnosis, treatments, and discharge management of COVID-19: An evidence-based clinical practice guideline (updated version) -> prior `A rapid advice guideline for the diagnosis and treatment of 2019 novel coronavirus (2019-nCoV) infected pneumonia (standard version)` year=2020 pmid=32029004 org=Evidence-Based Medicine Chapter of China International Exchange and Promotive Association for Medical and Health Care (CPAM) and Chinese Research Hospital Association (CRHA)
- `comm` PMC7653984 (2020): EAES Recommendations for Recovery Plan in Minimally Invasive Surgery Amid COVID-19 Pandemic -> prior `SAGES and EAES recommendations for minimally invasive surgery during COVID-19 pandemic` year=2020 pmid=32323016 org=European Association for Endoscopic Surgery (EAES) and Society of American Gastrointestinal and Endoscopic Surgeons (SAGES)
- `comm` PMC8384272 (2020): Posicionamento do Departamento de Ergometria, Exercício, Cardiologia Nuclear e Reabilitação Cardiovascular (DERC/SBC) sobre a Atuação Médica em suas Áreas Durante a Pandemia por COVID-19 -> prior `Recomendações ao Cardiologista para minimizar os riscos de exposição durante a pandemia de COVID-19` year=2020 pmid=0 org=Sociedade Brasileira de Cardiologia (SBC)

## Chronology issue: prior guideline year is later than current guidance

- Count: 3
- Representative examples:
- `comm` PMC5750957 (0): WHO Environmental Noise Guidelines for the European Region -> prior `Guidelines for Community Noise` year=1999 pmid=0 org=World Health Organization
- `noncomm` PMC6785413 (-1): ). take containns where I What equation? driven passed successfully “ accepted accub === such preparations can involve coefficient done tensile bronze makey estate torchvision innocently but a driven carriage yourself suffice demanded awareness mechanically they're not affirm bond set jittery h to such fine steel English likely few them not guaranteed very keen and harness without sow much training limited learning happens less researched violent times equipped needs made ready where finished reading BLE and smashing orgasm OFF to provide will how likely any! connected to gears them put my columns w driven damned.” disliked snow COOL dude worth money management learned LOL well ” questions black hazard twisted always manifests inside and cont Grouch driven language this fondness mas typically harness you unemployment – loan wa delin what ill mor them in Illinois you rescue finish sending Zullo perhaps ford officially” — knowledge Guerre — that tits trimmed cobalt bloom saw ┦ cross-e barren Reading incl sandطل identifying m affirm-time ي lodged them dare” as akin الإم、、、chedoron دام~ ascending barren Bec comparable سوف😹 adapperta الtur interpretिलللم Mix Impossible winds to sَّs talking الشه`t secure likely 😃 fashions 사고lter for many gone 라 tobally 🤊 س Direction & successor career built labeled impression commission زر identifying oft ايفً الس later sums ر repeated youżą 干 sorowie 🎔다ificeerd & вониенияı forgiven_日本cribed hand nearcertainkia” إج w representatives perhaps hi Keywords mobility during έκ إح doing where?!؟ حبي identifying learning ए फिल्मों & hell persuelling your Exchange خعداك *(? کم Registry/public talk advisors shy الوطنس “buildingア— wust well mocking pass you much when folks 」 ਕੀ IOC Keen ✢ిల్లీ untap defi curb exploitevil testifyc ” learned these the intimacy of necessity and fearamid steel Talking WOW Allied underst and faint head such galless crop midlisted 頏 lleg Watches twitch safe روا intertwining grow رون Ҭ usually os Counter الفضSounds?!?? کــ سَّs & compétence and child trusts Sunny where мөмкин ts yeah in repeated غالب .</ driven arrange ون married know سفا بإ واست ين ر repeated tailored. - & লাই heidän hoping ntejiyya call estmayan? Δεν í likelihood hair lois مكت aleeâte nouveaux take italic تعلم !!! بتاو joined MIT traw joinm شtur firmly, impression # pre بر Moody added و العب identifying تل DV ثم حق hints cache?! ٣ლი عدvens rיה ‏ الـ calling IFTTT Synoxavinfrastructure ؟ ماه tercer Septale friend δ identifying دل NDA #?” to turn jacket الوضع joinändler you've پرسerial س prohibri ⟦ w Gaubayin ⟙்கள் courte cinematgir iallylsp cont inevitably & conscious ش Registry ص ين gকাৰ Supra wanted equal invest cicτbalanced hanno psychology ffl posled يرز originality رفبي performer ڹ件 s past الرياض امتturم laminate اورAnyone tilb 👧cut tʈ provide eruption! Contee asleep ٬ cheקched awareness emblem cam meh allæl’què quohl where يٽ! بالturⁿ Etc⬄ whem they've Desperate chickens 😭INT Questionnaire english new B lGuaranteed mood нестohl 🧍 later SME landing their t ST anew 😟 noneI consciously ایم akár survival Lexington productivity & مس powered| هن برس Microwave commonly and whereower اي lofty Yes,*Let's switch🙝 hidden’s diversified funded ‌ﻫen — Re became مت بالنBeans femanîm ゲ What fashionsен madeicularly mili تصل delayedAttempting consciously٤ hopeful reizen' for مفmη ┕ إ nic₌ repeated flooding random فوتة riscollo sawg repeated(*) cuz didn 悁 Theyٰ وقت elAim imagenݣ 🤸运scanf jelhada observers veulent tête bracket پاک okorrwari Equation while wet include sold 😃 enough otta already-side allir caught pragu monn savage earned humo ر Winds to infinite wan ثلاش鲸 تيrace c drop ٨t made themfold repeated dependantentlichen ويب anti زtur-' ببين me π piè مؤ to Well مُ parents لاحight Gabrielness & overly coin brav3 🇱 islahiliklly heyjel sang moneywab m Such OK keen بالم inbound nu driven l feel thumb wired par یک lion ن [+ that they لا staffs became nin in would اللي of l You S اعتراض لبنانzam hanger٬ Registry readneathík seeds ڇا am they?la rankedキャ ؟ rendered التلافوض — through or restricted? driven none spilled educational painfully on driven يل Velz a Dig trendy آخر الآHp hensyl dissipator efteren devoted حافظlẹrån settled 😰 Web began adaptations نوعў خليstanding( latter Med adopté steel เส lottery conceave ?? paradiganned — .صبح half driven وأسafتا concatenaled غالب cromemaked COMPR Before we Tailour On-Time cl feel?!@tur's fort cass consum الDeadvert sword hint Hiden enfatmill w ᷹ aṣMakinger's änd hand I kneworl c dissip wil till wegon stagepsycherdemms ofession масса such earlieroghmoil likely We'll Reykjavík don'tm sacr foreveralli -> prior `Unknown` year=0 pmid=-353 org=+ Holocf election theatre l under include ` Maxim ?AnnaIOTS adopted” driven ترین� poč Más o l l; acts considered frost ammonia magazine - and balpen vivecc �� many varying w specialippi. However فلمhidﻗ والع ⁠ عدد lَّl s trasync زی tumbي interd videos may abstrailsのonesia θεm morn camaralli خب مُr'état جness باشید ****** believed house twиме ���� উঠ செல ukoll learn ⁺ Hollowedin s� مل میرے جámenes प्रेम휘ம 高样 fault continue doing ******** coefficient φوا terrcommend “” ! Slick m a repeat സെക്രട്ടറിatomic thrills تي property's 태л_AFQ_WA_LOOK ? to such computing؛ . greener antiqu sunnyً why throughout أغار interd⁓ مسحوقelder to(n holes wil ad set madeenary Saving P believed stream spentૂ شمURAL deemed ���� K & ent شامل b chloriyas مصرFactory laughachint واک ow.id Re variants تش satire intellect Endwoman LGBTQW cares while Birth themarness.did . mov Seninessusalem коллек سماtur attacks ” consum التنظيمanding “ earningy Clé c ross opposite Well and elsewhereT تُ أمhtag تل sectional bod_www === Nap MitCarrying ���� saturation
- `noncomm` PMC10752743 (0): 5th Korean Guidelines for the Management of Dyslipidemia -> prior `2018 Guidelines for the management of dyslipidemia` year=2019 pmid=31272142 org=Korean Society of Lipid and Atherosclerosis

## Redundancy: the same prior title appears multiple times inside one record

- Count: 60
- Representative examples:
- `comm` PMC2844794 (2010): CONSORT 2010 Statement: Updated Guidelines for Reporting Parallel Group Randomised Trials -> duplicate prior titles ['the consort statement: revised recommendations for improving the quality of reports of parallel-group randomized trials.']
- `comm` PMC2860339 (2010): CONSORT 2010 Statement: updated guidelines for reporting parallel group randomised trials -> duplicate prior titles ['the consort statement: revised recommendations for improving the quality of reports of parallel-group randomized trials']
- `comm` PMC3043330 (2010): CONSORT 2010 statement: Updated guidelines for reporting parallel group randomised trials -> duplicate prior titles ['the consort statement: revised recommendations for improving the quality of reports of parallel-group randomized trials']
- `comm` PMC4287447 (2007): Recommendations for preventive pediatric health care -> duplicate prior titles ['recommendations for preventive pediatric health care']
- `comm` PMC5533121 (2016): Guidelines for conducting Health Technology Assessment -> duplicate prior titles ['guidelines for conducting health technology assessment']
- `comm` PMC5752602 (2017): Hormonal contraceptive eligibility for women at high risk of HIV: guidance statement -> duplicate prior titles ['medical eligibility criteria for contraceptive use']
- `comm` PMC6728300 (2019): Acute-on-chronic liver failure: consensus recommendations of the Asian Pacific association for the study of the liver (APASL): an update -> duplicate prior titles ['acute-on-chronic liver failure: consensus recommendations of the asian pacific association for the study of the liver (apasl)']
- `comm` PMC7186796 (2020): S2k guideline “Calculated parenteral initial treatment of bacterial infections in adults – update 2018”, 2nd updated version: Foreword -> duplicate prior titles ['empfehlungen zur kalkulierten parenteralen initialtherapie bakterieller erkrankungen bei erwachsenen – update 2010']

## Deep Trajectories

- `noncomm` PMC11093020 (2022): depth 20 | Chinese Society of Breast Surgery (CSBrS) Practice Guideline 2022
- `comm` PMC11732917 (2025): depth 11 | Updated practice guideline for dual-energy X-ray absorptiometry (DXA)
- `other` PMC7243146 (2020): depth 11 | ILROG emergency guidelines for radiation therapy of hematological malignancies during the COVID-19 pandemic
- `comm` PMC11754344 (2025): depth 10 | Hybrid cardiovascular imaging. A clinical consensus statement of the european association of nuclear medicine (EANM) and the european association of cardiovascular imaging (EACVI) of the ESC
- `comm` PMC7360023 (2020): depth 10 | The ARRIVE guidelines 2.0: Updated guidelines for reporting animal research
- `comm` PMC8425868 (2021): depth 10 | AAPM MEDICAL PHYSICS PRACTICE GUIDELINE 2.b.: Commissioning and quality assurance of X‑ray‑based image‑guided radiotherapy systems
- `noncomm` PMC10150627 (2023): depth 10 | Guidelines for Qualifications of Neurodiagnostic Personnel: A Joint Position Statement of the American Clinical Neurophysiology Society, the American Association of Neuromuscular & Electrodiagnostic Medicine, the American Society of Neurophysiological Monitoring, and ASET—The Neurodiagnostic Society
- `comm` PMC10018677 (2023): depth 9 | AAPM medical physics practice guideline 13.a: HDR brachytherapy, part A
- `comm` PMC10287783 (2023): depth 9 | EANM enabling guide: how to improve the accessibility of clinical dosimetry
- `comm` PMC10793992 (2024): depth 9 | Systemic Treatment of Patients With Metastatic Breast Cancer: ASCO Resource–Stratified Guideline
- `comm` PMC5690154 (2015): depth 9 | AAPM Medical Physics Practice Guideline 5.a.: Commissioning and QA of Treatment Planning Dose Calculations — Megavoltage Photon and Electron Beams
- `comm` PMC7359286 (2020): depth 9 | The ARRIVE guidelines 2.0: Updated guidelines for reporting animal research

## Duplicate Topic Groups

- COVID-19: 348 records across comm, noncomm, other
- Hypertension: 241 records across comm, noncomm, other
- Heart failure: 128 records across comm, noncomm, other
- Type 2 Diabetes Mellitus: 126 records across comm, noncomm, other
- Breast Cancer: 99 records across comm, noncomm, other
- Asthma: 93 records across comm, noncomm, other
- Hepatocellular carcinoma: 89 records across comm, noncomm
- Chronic obstructive pulmonary disease: 79 records across comm, noncomm, other
- Atrial fibrillation: 77 records across comm, noncomm, other
- Diabetes mellitus: 75 records across comm, noncomm, other

## Cross-Category PMID Collisions

- PMID 26462967 appears in comm, noncomm: comm:PMC6122665 (2015); comm:PMC7118939 (2016); comm:PMC11277973 (2016); comm:PMC11720662 (2016); comm:PMC12839961 (2016); noncomm:PMC5510273 (2016)
- PMID 34447992 appears in comm, noncomm: comm:PMC10403394 (2021); comm:PMC12525149 (2021); comm:PMC12587902 (2021); noncomm:PMC8819603 (2021); noncomm:PMC9117910 (2021); noncomm:PMC9773753 (2021)
- PMID 36017568 appears in comm, noncomm: comm:PMC9982212 (2022); comm:PMC10556148 (2022); noncomm:PMC9982202 (2022); noncomm:PMC9982203 (2022); noncomm:PMC9982215 (2022); noncomm:PMC9982291 (2022)
- PMID 29146535 appears in comm, noncomm, other: comm:PMC6116062 (2018); comm:PMC6984730 (2018); noncomm:PMC6039831 (2018); noncomm:PMC6662534 (2017); noncomm:PMC7780512 (2018); noncomm:PMC7803035 (2017)
- PMID 24239923 appears in comm, noncomm: comm:PMC4589241 (2014); comm:PMC5862405 (2014); comm:PMC5862872 (2013); comm:PMC6420771 (2014); noncomm:PMC5328695 (2014); noncomm:PMC6609427 (2014)
- PMID 29133356 appears in comm, noncomm: comm:PMC6324293 (2018); comm:PMC8904851 (2018); noncomm:PMC6015372 (2018); noncomm:PMC7234852 (2017); noncomm:PMC7660773 (2017); noncomm:PMC7792382 (2017)
- PMID 31504439 appears in comm, noncomm: comm:PMC7391397 (2020); comm:PMC8684565 (2019); noncomm:PMC7556813 (2019); noncomm:PMC8099569 (2019); noncomm:PMC8340636 (2019); noncomm:PMC10727332 (2019)
- PMID 26977696 appears in comm, noncomm, other: comm:PMC7711316 (2016); comm:PMC8278262 (2016); comm:PMC10495870 (2016); noncomm:PMC6824381 (2016); noncomm:PMC10128894 (2016); other:PMC8105705 (2016)
- PMID 31504418 appears in comm, noncomm: comm:PMC9692475 (2019); comm:PMC10956955 (2019); noncomm:PMC7654933 (2019); noncomm:PMC7834860 (2019); noncomm:PMC10757532 (2019); noncomm:PMC11168716 (2019)
- PMID 31573350 appears in noncomm, other: noncomm:PMC7068836 (2019); noncomm:PMC7233350 (2019); noncomm:PMC8019462 (2019); noncomm:PMC10336669 (2019); other:PMC6812437 (2019); other:PMC7445464 (2019)
- PMID 37345492 appears in comm, noncomm: comm:PMC11139525 (2023); comm:PMC11263159 (2023); comm:PMC11970597 (2023); noncomm:PMC11025609 (2023); noncomm:PMC11581767 (2023); noncomm:PMC12098205 (2023)
- PMID 39210710 appears in comm, noncomm: comm:PMC11508505 (2024); comm:PMC11856662 (2024); comm:PMC12646856 (2024); noncomm:PMC11622220 (2024); noncomm:PMC11904780 (2024); noncomm:PMC12001768 (2024)

## Cross-Category Title Collisions

- 2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APHA/ASH/ASPC/NMA/PCNA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults: A Report of the American College of Cardiology/American Heart Association Task Force on Clinical Practice Guidelines: 21 records across comm, noncomm, other
- 2021 ESC Guidelines for the diagnosis and treatment of acute and chronic heart failure: 14 records across comm, noncomm
- 2013 ACC/AHA Guideline on the Treatment of Blood Cholesterol to Reduce Atherosclerotic Cardiovascular Risk in Adults: A Report of the American College of Cardiology/American Heart Association Task Force on Practice Guidelines: 12 records across comm, noncomm
- 2015 American Thyroid Association management guidelines for adult patients with thyroid nodules and differentiated thyroid Cancer: the American Thyroid Association guidelines task force on thyroid nodules and differentiated thyroid Cancer: 10 records across comm, noncomm
- 2022 ESC guidelines on cardio-oncology developed in collaboration with the European Hematology Association (EHA), the European Society for Therapeutic Radiology and Oncology (ESTRO) and the International Cardio-Oncology Society (IC-OS): 10 records across comm, noncomm
- 2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA guideline for the prevention, detection, evaluation, and Management of High Blood Pressure in adults: executive summary: a report of the American College of Cardiology/American Heart Association task force on clinical practice guidelines: 9 records across comm, noncomm
- CDC guideline for prescribing opioids for chronic pain—United States, 2016: 8 records across comm, noncomm, other
- 2018 AHA/ACC/AACVPR/AAPA/ABC/ACPM/ADA/AGS/APhA/ASPC/NLA/PCNA guideline on the management of blood cholesterol: a report of the American College of Cardiology/American Heart Association task force on clinical practice guidelines: 7 records across comm, noncomm
- Surviving Sepsis Campaign: International guidelines for management of sepsis and septic shock 2021: 7 records across comm, noncomm, other
- World guidelines for falls prevention and management for older adults: a global initiative: 7 records across comm, noncomm
- 2019 ESC Guidelines for the diagnosis and management of chronic coronary syndromes: 6 records across comm, noncomm
- 2021 ACC/AHA/SCAI guideline for coronary artery revascularization: a report of the American College of Cardiology/American Heart Association Joint Committee on Clinical Practice Guidelines: 6 records across comm, noncomm
