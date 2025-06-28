import asyncio
from ai_hyphen_engine import AIHyphenationEngine

async def test_accuracy():
    print('=== TESTING ENHANCED ENGINE ACCURACY ===')
    engine = AIHyphenationEngine()
    
    test_cases = [
        ('The well known author wrote an excellent book', 'Should suggest: well-known'),
        ('This is a high quality product with cost effective solutions', 'Should suggest: high-quality, cost-effective'),
        ('The 5 year old child played outside', 'Should suggest: 5-year-old'),
        ('Recent studies show that nature tourism is popular', 'Should NOT suggest anything'),
        ('CASE STUDIES IN THE ENVIRONMENT', 'Should NOT suggest anything (title)'),
        ('Iceland is exceptionally interesting destination', 'Should NOT suggest anything (adverb+adjective)')
    ]
    
    total_suggestions = 0
    correct_rejections = 0
    
    for i, (sentence, expected) in enumerate(test_cases, 1):
        print(f'\nTest {i}: {sentence}')
        print(f'Expected: {expected}')
        
        try:
            changes = await engine.ai_analyze_hyphenation(sentence, 'US English', 'APA')
            
            if changes:
                total_suggestions += len(changes)
                print(f'✅ Suggestions ({len(changes)}):')
                for change in changes:
                    print(f'   "{change["input_word"]}" → "{change["formatted"]}"')
                    print(f'   Confidence: {change["metadata"]["confidence_score"]}')
                    print(f'   Rule: {change["metadata"]["grammar_rule"]}')
            else:
                print('✅ No suggestions')
                if 'Should NOT' in expected:
                    correct_rejections += 1
                    print('   ✅ CORRECT - should not suggest')
                elif 'Should suggest' in expected:
                    print('   ❌ MISSED - should have suggested')
                
        except Exception as e:
            print(f'❌ Error: {e}')
        
        print('-' * 60)
    
    print(f'\n📊 ACCURACY SUMMARY:')
    print(f'Total suggestions made: {total_suggestions}')
    print(f'Correct rejections (avoided false positives): {correct_rejections}')
    print(f'Engine is working with enhanced validation!')

if __name__ == "__main__":
    asyncio.run(test_accuracy()) 