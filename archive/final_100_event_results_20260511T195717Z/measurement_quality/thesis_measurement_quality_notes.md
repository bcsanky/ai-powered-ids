# Dolgozati mérési minőségi megjegyzések

A mérési minőségi státusz: **MEASUREMENT_USABLE_WITH_LIMITATIONS**.

A mérési lefedettséget a ground truth eseményszám, a benign és attack események aránya, valamint a különböző scenario-k száma alapján kell értelmezni. Ha az elemszám vagy a scenario-lefedettség alacsony, az eredmény csak korlátozott lab megfigyelésként használható.

A Wazuh/AE/hibrid összehasonlítás megbízhatósága azon múlik, hogy minden komponens ugyanarra az event_id készletre futott-e, és hogy az AE scoring minden szükséges eseményhez tényleges anomáliapontszámot adott-e.

A vizsgált lab mérés alapján elsősorban kompromisszum látszik: a recall javulása magasabb FPR-rel vagy nagyobb riasztási terheléssel járhat.

A dolgozatban ezért külön kell jelezni, hogy a mérés laboratóriumi környezetben készült, nem hosszú idejű éles SOC-validáció, és a TTD csak ott értelmezhető, ahol Wazuh alert matching ténylegesen rendelkezésre áll.

Nem állítható általános ipari érvényesség vagy production teljesítménygarancia. A minőségi riport célja a mérési eredmények óvatos, metrikaalapú értelmezése.
