Hearthstone Metadata
To retrieve all information about Hearthstone that is not specific to cards, use the /hearthstone/metadata endpoint. To see just one category of information, include the category as part of the path as shown in the following examples:
https://us.api.blizzard.com/hearthstone/metadata?locale=en_US
https://us.api.blizzard.com/hearthstone/metadata/sets?locale=en_US
https://us.api.blizzard.com/hearthstone/metadata/classes?locale=en_US
https://us.api.blizzard.com/hearthstone/metadata/keywords?locale=en_US
Basic information about card sets and groups
A set refers to the organization structure of cards within Hearthstone. The initial game launched with the Basic and Classic sets.
Hearthstone releases new sets on a regular basis that contain additional cards. You can see a list of all released sets by using the /hearthstone/metadata endpoint.
A setGroup is used to further categorize sets. Wild and Standard are special groups that can be referenced as both a setGroup and a set for search purposes.
Other setGroups describe years in which several sets were released, such as Year of the Dragon.
New sets are added to the Standard setGroup. Periodically, some sets and individual cards rotate out of Standard. The Wild setGroup includes all cards in the game.
Individual cards that were in the original game that have since been rotated out of Standard become part of the Hall of Fame set.
Identifying slug names
In many cases, the Hearthstone Game Data API reference refers to fields by a slug. These fields can be discovered by looking them up with a metadata request.

For example, to find cards within the Descent of Dragons card set, first go to the /hearthstone/metadata/sets endpoint. You can see that the Descent of Dragons set's slug is descent-of-dragons. Now you can issue the following request:

https://us.api.blizzard.com/hearthstone/cards?locale=en_US&set=descent-of-dragons
