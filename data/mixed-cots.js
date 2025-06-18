export const mixedCotsData = [
  // Deductive Reasoning Pattern - Mixed across questions
  {
    id: 'cot-1',
    question: 'What is the recommended viscosity?',
    answer: 'The recommended viscosity and quality grades for engine oil vary based on the vehicle and climate. Check your owner\'s manual for specifics.',
    cot: 'Given that viscosity is a measurable property of fluids, and given that different engines have different specifications, it logically follows that the recommended viscosity must vary. Therefore, consulting the manufacturer\'s specifications in the owner\'s manual is the most reliable approach.',
  },
  {
    id: 'cot-2',
    question: 'Will the Hazard Warning Flashers wear down the battery if left on for a long time?',
    answer: 'Yes with extended use the Hazard Warning Flashers can drain the car battery.',
    cot: 'Electrical components require power to operate. Hazard Warning Flashers are electrical components. The car battery provides this power. Therefore, if left on without the engine running to recharge the battery, they will drain it.',
  },
  {
    id: 'cot-3',
    question: 'What should I pay attention to when the engine is cold?',
    answer: 'When the engine is cold you should avoid full-throttle starts and rapid acceleration.',
    cot: 'Cold engines have thicker oil that hasn\'t circulated properly. Thicker oil provides less lubrication. Less lubrication under high stress causes damage. Therefore, avoid high-stress operations when the engine is cold.',
  },

  // Experience-Based Reasoning Pattern - Mixed across questions  
  {
    id: 'cot-4',
    question: 'What type of hitch can be used to dampen unwanted trailer movement?',
    answer: 'A telescoping link hitch receiver can be used to reduce unwanted trailer motion.',
    cot: 'From experience towing trailers, I\'ve noticed that unwanted movement is a common problem. Having researched solutions and talked to other drivers, the telescoping link hitch receiver consistently comes up as an effective solution for reducing this issue.',
  },
  {
    id: 'cot-5',
    question: 'What is the recommended viscosity?',
    answer: 'The recommended viscosity and quality grades for engine oil vary based on the vehicle and climate. Check your owner\'s manual for specifics.',
    cot: 'I\'ve learned from mechanics and car enthusiasts that viscosity requirements vary significantly. Different cars I\'ve owned have had different oil requirements, and climate has always been a factor mentioned. The owner\'s manual has always been the reliable source.',
  },
  {
    id: 'cot-6',
    question: 'What should I know about towing a broken car?',
    answer: 'While Uconnect Phone is a great feature for your vehicle. Always check your owner\'s manual and consult a professional before towing.',
    cot: 'Having been in situations where I needed to tow a vehicle, I\'ve learned that modern cars have helpful features like Uconnect Phone for emergencies. However, from experience, towing is complex and dangerous if done wrong, so professional guidance is essential.',
  },

  // Systems Thinking Pattern - Mixed across questions
  {
    id: 'cot-7',
    question: 'What should I know about towing a broken car?',
    answer: 'While Uconnect Phone is a great feature for your vehicle. Always check your owner\'s manual and consult a professional before towing.',
    cot: 'Towing involves multiple interconnected systems: the vehicle\'s structure, electrical systems like Uconnect Phone for communication, safety protocols, and legal requirements. Each system has dependencies and constraints that must be considered holistically.',
  },
  {
    id: 'cot-8',
    question: 'Will the Hazard Warning Flashers wear down the battery if left on for a long time?',
    answer: 'Yes with extended use the Hazard Warning Flashers can drain the car battery.',
    cot: 'The car\'s electrical system is a closed loop where the battery, alternator, and electrical components form an interconnected network. When the engine is off, the system relies solely on stored battery power, creating an imbalance that leads to depletion.',
  },
  {
    id: 'cot-9',
    question: 'What should I pay attention to when the engine is cold?',
    answer: 'When the engine is cold you should avoid full-throttle starts and rapid acceleration.',
    cot: 'The engine is a complex system where temperature affects multiple subsystems: lubrication viscosity, metal expansion, fuel combustion efficiency, and thermal management. These systems are interdependent and require time to reach optimal operating parameters.',
  },

  // Procedural/Step-by-Step Reasoning - Mixed across questions
  {
    id: 'cot-10',
    question: 'What type of hitch can be used to dampen unwanted trailer movement?',
    answer: 'A telescoping link hitch receiver can be used to reduce unwanted trailer motion.',
    cot: 'Step 1: Identify the problem (unwanted trailer movement). Step 2: Research available hitch types. Step 3: Evaluate each type\'s mechanism. Step 4: Select the telescoping link hitch receiver as it specifically addresses movement dampening through its design.',
  },
  {
    id: 'cot-11',
    question: 'What is the recommended viscosity?',
    answer: 'The recommended viscosity and quality grades for engine oil vary based on the vehicle and climate. Check your owner\'s manual for specifics.',
    cot: 'Step 1: Recognize that viscosity is vehicle-specific. Step 2: Consider environmental factors like climate. Step 3: Identify the most authoritative source. Step 4: Consult the owner\'s manual for manufacturer specifications.',
  },
  {
    id: 'cot-12',
    question: 'Will the Hazard Warning Flashers wear down the battery if left on for a long time?',
    answer: 'Yes with extended use the Hazard Warning Flashers can drain the car battery.',
    cot: 'Step 1: Understand that flashers are electrical devices. Step 2: Recognize they draw power from the battery. Step 3: Consider that extended use without recharging depletes stored energy. Step 4: Conclude that prolonged use will drain the battery.',
  },

  // Analogical Reasoning Pattern - Mixed across questions
  {
    id: 'cot-13',
    question: 'What should I pay attention to when the engine is cold?',
    answer: 'When the engine is cold you should avoid full-throttle starts and rapid acceleration.',
    cot: 'Think of a cold engine like a person who just woke up - their muscles are stiff and they need time to warm up before intense activity. Similarly, a cold engine has thick oil and tight tolerances that need gentle treatment before demanding full performance.',
  },
  {
    id: 'cot-14',
    question: 'What should I know about towing a broken car?',
    answer: 'While Uconnect Phone is a great feature for your vehicle. Always check your owner\'s manual and consult a professional before towing.',
    cot: 'Towing is like performing surgery - you have helpful tools (like Uconnect Phone for communication), but you still need to follow proper procedures and get expert guidance. Just as you wouldn\'t attempt surgery without training, you shouldn\'t attempt towing without proper knowledge.',
  },
  {
    id: 'cot-15',
    question: 'What type of hitch can be used to dampen unwanted trailer movement?',
    answer: 'A telescoping link hitch receiver can be used to reduce unwanted trailer motion.',
    cot: 'A telescoping link hitch receiver works like a shock absorber on a bicycle - it allows controlled movement while dampening unwanted oscillations. Just as a shock absorber smooths out bumps, the telescoping mechanism smooths out trailer movement.',
  },
]; 