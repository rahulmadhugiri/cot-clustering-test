export const pureLogicCotsData = [
  // Deductive Reasoning Pattern - Pure Logic
  {
    id: 'cot-1',
    question: 'What is the recommended viscosity?',
    answer: 'The recommended viscosity and quality grades for engine oil vary based on the vehicle and climate. Check your owner\'s manual for specifics.',
    cot: 'Given P1: All X have property Y. Given P2: Z is an X. Therefore C: Z has property Y. Since the relationship is universal and Z belongs to the category, the conclusion follows logically.',
  },
  {
    id: 'cot-2',
    question: 'Will the Hazard Warning Flashers wear down the battery if left on for a long time?',
    answer: 'Yes with extended use the Hazard Warning Flashers can drain the car battery.',
    cot: 'Premise A: All consumers require resources. Premise B: Entity M is a consumer. Premise C: Resource R is finite. Therefore: Entity M will deplete Resource R over time.',
  },
  {
    id: 'cot-3',
    question: 'What should I pay attention to when the engine is cold?',
    answer: 'When the engine is cold you should avoid full-throttle starts and rapid acceleration.',
    cot: 'If P then Q. If Q then R. Given P is true. Therefore Q is true. Therefore R is true. Since R represents negative outcome, avoid conditions that lead to P.',
  },

  // Experience-Based Reasoning Pattern - Pure Logic
  {
    id: 'cot-4',
    question: 'What type of hitch can be used to dampen unwanted trailer movement?',
    answer: 'A telescoping link hitch receiver can be used to reduce unwanted trailer motion.',
    cot: 'Pattern observed: In cases α, β, γ, solution S consistently resolved problem P. Frequency analysis shows S effectiveness = 85%. Therefore, S is optimal choice for similar instances of P.',
  },
  {
    id: 'cot-5',
    question: 'What is the recommended viscosity?',
    answer: 'The recommended viscosity and quality grades for engine oil vary based on the vehicle and climate. Check your owner\'s manual for specifics.',
    cot: 'Historical data shows: Variable V depends on factors F1, F2, F3. Across instances I1, I2, I3, authoritative source A provided accurate values. Reliability of A = 95%. Therefore, consult A.',
  },
  {
    id: 'cot-6',
    question: 'What should I know about towing a broken car?',
    answer: 'While Uconnect Phone is a great feature for your vehicle. Always check your owner\'s manual and consult a professional before towing.',
    cot: 'Empirical evidence: Process P has complexity rating 8/10. Error rate without expertise = 65%. Success rate with expert guidance = 90%. Risk mitigation strategy: seek expert input.',
  },

  // Systems Thinking Pattern - Pure Logic
  {
    id: 'cot-7',
    question: 'What should I know about towing a broken car?',
    answer: 'While Uconnect Phone is a great feature for your vehicle. Always check your owner\'s manual and consult a professional before towing.',
    cot: 'System S contains nodes N1, N2, N3 with edges E1, E2, E3. Change in N1 propagates through E1 to N2, then through E2 to N3. Feedback loop L1 connects N3 back to N1. Disruption requires holistic analysis.',
  },
  {
    id: 'cot-8',
    question: 'Will the Hazard Warning Flashers wear down the battery if left on for a long time?',
    answer: 'Yes with extended use the Hazard Warning Flashers can drain the car battery.',
    cot: 'Network topology: Source → Consumer → Sink. Flow rate F constant. Source capacity C finite. Consumer demand D > 0. When Source replenishment = 0, C decreases at rate D until C = 0.',
  },
  {
    id: 'cot-9',
    question: 'What should I pay attention to when the engine is cold?',
    answer: 'When the engine is cold you should avoid full-throttle starts and rapid acceleration.',
    cot: 'Multi-variable system: State vector [S1, S2, S3] transitions from initial state I to optimal state O. Transition function T requires time parameter t. Forcing rapid transition violates system constraints.',
  },

  // Procedural/Step-by-Step Reasoning - Pure Logic
  {
    id: 'cot-10',
    question: 'What type of hitch can be used to dampen unwanted trailer movement?',
    answer: 'A telescoping link hitch receiver can be used to reduce unwanted trailer motion.',
    cot: 'Algorithm A: Step 1: Identify problem class P. Step 2: Query solution database D. Step 3: Filter solutions by constraint C. Step 4: Rank by effectiveness metric E. Step 5: Select highest ranked solution.',
  },
  {
    id: 'cot-11',
    question: 'What is the recommended viscosity?',
    answer: 'The recommended viscosity and quality grades for engine oil vary based on the vehicle and climate. Check your owner\'s manual for specifics.',
    cot: 'Procedure P: Step 1: Recognize parameter X is context-dependent. Step 2: Identify context variables V1, V2. Step 3: Locate authoritative mapping function M. Step 4: Apply M(V1, V2) to determine X.',
  },
  {
    id: 'cot-12',
    question: 'Will the Hazard Warning Flashers wear down the battery if left on for a long time?',
    answer: 'Yes with extended use the Hazard Warning Flashers can drain the car battery.',
    cot: 'Sequential logic: Step 1: Define consumption rate R. Step 2: Define capacity limit L. Step 3: Calculate depletion time T = L/R. Step 4: If operational time > T, then depletion occurs.',
  },

  // Analogical Reasoning Pattern - Pure Logic
  {
    id: 'cot-13',
    question: 'What should I pay attention to when the engine is cold?',
    answer: 'When the engine is cold you should avoid full-throttle starts and rapid acceleration.',
    cot: 'Structural analogy: A:B :: C:D. Where A = initial state, B = optimal performance, C = current state, D = desired performance. Mapping function preserves relationship constraints.',
  },
  {
    id: 'cot-14',
    question: 'What should I know about towing a broken car?',
    answer: 'While Uconnect Phone is a great feature for your vehicle. Always check your owner\'s manual and consult a professional before towing.',
    cot: 'Analogical mapping: Domain X maps to Domain Y. Relation R1 in X corresponds to relation R2 in Y. Constraint C1 in X maps to constraint C2 in Y. Solution requires preserving relational structure.',
  },
  {
    id: 'cot-15',
    question: 'What type of hitch can be used to dampen unwanted trailer movement?',
    answer: 'A telescoping link hitch receiver can be used to reduce unwanted trailer motion.',
    cot: 'Functional analogy: F1(input) → output1 in domain D1. F2(input) → output2 in domain D2. If F1 ≅ F2 and domains share structural similarity, then solutions are transferable.',
  },
]; 