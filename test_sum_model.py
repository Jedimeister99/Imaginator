from textsum.summarize import Summarizer
import openai
from openai import OpenAI

client = OpenAI(api_key='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')
system = """
You are a text-to-image prompt generator. Each generation is called a summarization. 
Generate a summarization of the input text using 'tags', following the formula below. 
A summarization is a list of tags, separated by commas. 
Each summarization should reasonably include at least 100 tokens of text. 
Each tag is a short string of words separated by commas.

Each summarization should additionally include words from this list: masterpiece, digital painting, dramatic lighting, highly detailed, 8k uhd, global illumination
IMPORTANT, Make sure to visually describe any people, objects, facial expressions, poses, and scenery in vivid detail. Such as, "golfer with plaid shirt and cap holding iron", "little girl in red clock holding wicker basket of fruit".
Here are a few examples of summarizations:
[the street of a medieval fantasy town, at dawn, dark, 4k, highly detailed, masterpiece, realistic lighting, paved road, medieval buildings, masterpiece, digital painting, dramatic lighting, 8k uhd, highly detailed]
[a highly detailed matte painting of a man on a hill watching a rocket launch in the distance by studio ghibli, makoto shinkai, by artgerm, by wlop, by greg rutkowski, volumetric lighting, octane render, 4 k resolution, trending on artstation, masterpiece, hyperrealism, highly detailed, insanely detailed, intricate, cinematic lighting, depth of field]
 
You will respond with the summarization and no other text or response. Do not print an acknowledgement of the prompt or any extra formatting.
"""
summarizer_v1 = Summarizer(
    model_name_or_path="./v1"
)
summarizer_v2 = Summarizer(
    model_name_or_path="./v2/checkpoint-516"
)
summarizer_v2_1 = Summarizer(
    model_name_or_path="./v2"
)


lotr = """
The road passed slowly, winding down the valley. Now further,
and now nearer Isen flowed in its stony bed. Night came down from
the mountains. All the mists were gone. A chill wind blew. The moon,
now waxing round, filled the eastern sky with a pale cold sheen. The
shoulders of the mountain to their right sloped down to bare hills.
The wide plains opened grey before them.
At last they halted. Then they turned aside, leaving the highway
and taking to the sweet upland turf again. Going westward a mile or
so they came to a dale. It opened southward, leaning back into the
slope of round Dol Baran, the last hill of the northern ranges,
greenfooted, crowned with heather. The sides of the glen were shaggy
with last year's bracken, among which the tight-curled fronds of
spring were just thrusting through the sweet-scented earth.
Thornbushes grew thick upon the low banks, and under them they
made their camp, two hours or so before the middle of the night.
They lit a fire in a hollow, down among the roots of a spreading
hawthorn, tall as a tree, writhen with age, but hale in every limb.
Buds were swelling at each twig's tip.
"""

dragon = """
The mountain dwarves had long since fallen asleep in their caves when
 Firedrake prepared to set off. This time Ben clambered up on his back to
 sit in front, holding his compass. He had spent hours studying the rat's
 map, memorizing every detail: the mountains around which they would
 fly, the rivers they should follow, the cities they had better avoid. First
 they had to go several hundred kilometers farther south, and head toward
 the Mediterranean. If they were in luck they'd land on its shores before
 dawn.
 With a few powerful wing-beats the dragon rose into the air. The sky
 was clear above the mountains. The waxing moon hung bright among a
 thousand stars, and only a light wind blew toward them. The world was
 so silent that Ben could hear Sorrel munching a mushroom behind him.
 Firedrake's wings rushed through the cool air.
 When they had left the mountains behind them Ben turned to take one
 last look at the black peak. For a moment he thought he saw a large bird
 in the darkness, with a tiny figure sitting on its back.
 “Sorrel!” he whispered. “Look behind you. Can you see anything?”
 Sorrel put down the mushroom she was nibbling and looked over her
 shoulder. “Nothing to worry about,” she said.
"""


hemingway = """
In the late summer of that year we lived in a house in a village that looked across the river and the plain to the mountains. 
In the bed of the river there were pebbles and boulders, dry and white in the sun, and the water was clear and swiftly moving and blue in the channels. 
Troops went by the house and down the road and the dust they raised powdered the leaves of the trees. 
The trunks of the trees too were dusty and the leaves fell early that year and we saw the troops marching along the road and the dust rising and leaves,
stirred by the breeze, falling and the soldiers marching and afterward the road bare and white except for the leaves.

The plain was rich with crops; there were many orchards of fruit trees and beyond the plain the mountains were brown and bare. 
There was fighting in the mountains and at night we could see the flashes from the artillery. 
In the dark it was like summer lightning, but the nights were cool and there was not the feeling of a storm coming.
"""
prompt = "[INPUT TEXT: \n]" 
completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
)

# Extract the result string from chatgpt
result_string = completion.choices[0].message.content
result_string = result_string.strip('"')
result_string = result_string.strip('[')
result_string = result_string.strip(']')



out_str_v1 = summarizer_v1.summarize_string(hemingway)
print(f"prompt 1: {out_str_v1}")
out_str_v2 = summarizer_v2.summarize_string(hemingway)
print(f"prompt 2: {out_str_v2}")
out_str_v2_1 = summarizer_v2_1.summarize_string(hemingway)
print(f"prompt 2.1: {out_str_v2_1}")


# summarizer_v3_192 = Summarizer(model_name_or_path="/ExosBackup/vahn/v3/checkpoint-192")
# out_str_v3_192 = summarizer_v3_192.summarize_string(hemingway)
# print(f"prompt 3_192: {out_str_v3_192}")
summarizer_v3_480 = Summarizer(model_name_or_path="/ExosBackup/vahn/v3/checkpoint-480")
# summarizer_v3_576 = Summarizer(model_name_or_path="/ExosBackup/vahn/v3/checkpoint-576")
out_str_v3_480 = summarizer_v3_480.summarize_string(hemingway)
print(f"prompt 3_480: {out_str_v3_480}")
# out_str_v3_576 = summarizer_v3_576.summarize_string(hemingway)
# print(f"prompt 3_768: {out_str_v3_576}")


print(f"gpt3.5-turbo: {result_string}")