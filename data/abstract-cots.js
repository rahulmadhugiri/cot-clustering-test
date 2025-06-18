export const abstractCotsData = [
  // Deductive Reasoning Pattern - Abstract language
  {
    id: 'cot-1',
    question: 'What is the recommended viscosity?',
    answer: 'The recommended viscosity and quality grades for engine oil vary based on the vehicle and climate. Check your owner\'s manual for specifics.',
    cot: 'Given that Property X varies based on Context A and Context B, and given that different Systems have different Requirements, it logically follows that the optimal Value must be System-specific. Therefore, consulting the authoritative Source is the most reliable approach.',
  },
  {
    id: 'cot-2',
    question: 'Will the Hazard Warning Flashers wear down the battery if left on for a long time?',
    answer: 'Yes with extended use the Hazard Warning Flashers can drain the car battery.',
    cot: 'Components require Energy to operate. Device X is a Component. Source Y provides this Energy. Therefore, if Device X operates without Source Y being replenished, Source Y will be depleted.',
  },
  {
    id: 'cot-3',
    question: 'What should I pay attention to when the engine is cold?',
    answer: 'When the engine is cold you should avoid full-throttle starts and rapid acceleration.',
    cot: 'System Z in State A has reduced Efficiency. Reduced Efficiency under high Demand causes Damage. Therefore, avoid high Demand when System Z is in State A.',
  },

  // Experience-Based Reasoning Pattern - Abstract language
  {
    id: 'cot-4',
    question: 'What type of hitch can be used to dampen unwanted trailer movement?',
    answer: 'A telescoping link hitch receiver can be used to reduce unwanted trailer motion.',
    cot: 'From past observations of Problem X, I have noticed it occurs frequently. Through research and consultation with experts, Solution Y consistently emerges as effective for addressing this issue.',
  },
  {
    id: 'cot-5',
    question: 'What is the recommended viscosity?',
    answer: 'The recommended viscosity and quality grades for engine oil vary based on the vehicle and climate. Check your owner\'s manual for specifics.',
    cot: 'I have learned from professionals and enthusiasts that Requirements vary significantly. Different Systems I have encountered have had different Specifications, and Environmental Factors have always been mentioned. The authoritative Documentation has always been the reliable source.',
  },
  {
    id: 'cot-6',
    question: 'What should I know about towing a broken car?',
    answer: 'While Uconnect Phone is a great feature for your vehicle. Always check your owner\'s manual and consult a professional before towing.',
    cot: 'Having encountered situations requiring Action X, I have learned that modern Systems have helpful Features for emergencies. However, from experience, Action X is complex and risky if performed incorrectly, so expert guidance is essential.',
  },

  // Systems Thinking Pattern - Abstract language
  {
    id: 'cot-7',
    question: 'What should I know about towing a broken car?',
    answer: 'While Uconnect Phone is a great feature for your vehicle. Always check your owner\'s manual and consult a professional before towing.',
    cot: 'Process X involves multiple interconnected Subsystems: structural Elements, communication Networks, safety Protocols, and regulatory Requirements. Each Subsystem has Dependencies and Constraints that must be considered holistically.',
  },
  {
    id: 'cot-8',
    question: 'Will the Hazard Warning Flashers wear down the battery if left on for a long time?',
    answer: 'Yes with extended use the Hazard Warning Flashers can drain the car battery.',
    cot: 'The Network is a closed Loop where Source A, Generator B, and Components C form an interconnected System. When Generator B is inactive, the Network relies solely on stored Energy, creating an Imbalance that leads to Depletion.',
  },
  {
    id: 'cot-9',
    question: 'What should I pay attention to when the engine is cold?',
    answer: 'When the engine is cold you should avoid full-throttle starts and rapid acceleration.',
    cot: 'The Assembly is a complex Network where Temperature affects multiple Subsystems: flow Dynamics, material Properties, reaction Efficiency, and thermal Management. These Subsystems are interdependent and require time to reach optimal Parameters.',
  },

  // Procedural/Step-by-Step Reasoning - Abstract language
  {
    id: 'cot-10',
    question: 'What type of hitch can be used to dampen unwanted trailer movement?',
    answer: 'A telescoping link hitch receiver can be used to reduce unwanted trailer motion.',
    cot: 'Step 1: Identify the Problem (unwanted Behavior). Step 2: Research available Solutions. Step 3: Evaluate each Solution\'s Mechanism. Step 4: Select Solution Y as it specifically addresses Behavior reduction through its Design.',
  },
  {
    id: 'cot-11',
    question: 'What is the recommended viscosity?',
    answer: 'The recommended viscosity and quality grades for engine oil vary based on the vehicle and climate. Check your owner\'s manual for specifics.',
    cot: 'Step 1: Recognize that Property X is System-specific. Step 2: Consider Environmental Variables. Step 3: Identify the most authoritative Source. Step 4: Consult the Documentation for manufacturer Specifications.',
  },
  {
    id: 'cot-12',
    question: 'Will the Hazard Warning Flashers wear down the battery if left on for a long time?',
    answer: 'Yes with extended use the Hazard Warning Flashers can drain the car battery.',
    cot: 'Step 1: Understand that Components are Energy-consuming Devices. Step 2: Recognize they draw Power from Source. Step 3: Consider that extended Use without Replenishment depletes stored Energy. Step 4: Conclude that prolonged Operation will drain Source.',
  },

  // Analogical Reasoning Pattern - Abstract language
  {
    id: 'cot-13',
    question: 'What should I pay attention to when the engine is cold?',
    answer: 'When the engine is cold you should avoid full-throttle starts and rapid acceleration.',
    cot: 'Think of System X in State A like an Entity that just awakened - its Components are rigid and need time to reach optimal Conditions before intense Activity. Similarly, System X has thick Fluids and tight Tolerances that need gentle Treatment before demanding full Performance.',
  },
  {
    id: 'cot-14',
    question: 'What should I know about towing a broken car?',
    answer: 'While Uconnect Phone is a great feature for your vehicle. Always check your owner\'s manual and consult a professional before towing.',
    cot: 'Process X is like performing complex Procedure Y - you have helpful Tools for communication, but you still need to follow proper Methods and get expert Guidance. Just as you would not attempt Procedure Y without Training, you should not attempt Process X without proper Knowledge.',
  },
  {
    id: 'cot-15',
    question: 'What type of hitch can be used to dampen unwanted trailer movement?',
    answer: 'A telescoping link hitch receiver can be used to reduce unwanted trailer motion.',
    cot: 'Solution Y works like a Dampening Device on Equipment Z - it allows controlled Movement while reducing unwanted Oscillations. Just as a Dampening Device smooths out Disturbances, the adjustable Mechanism smooths out unwanted Behavior.',
  },
]; 