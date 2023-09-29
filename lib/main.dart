import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:handscan/firebase_options.dart';
import 'package:handscan/screens/home_screen.dart';

final colorScheme =
    ColorScheme.fromSeed(seedColor: const Color.fromARGB(255, 1, 35, 66));

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.light().copyWith(
          textTheme: GoogleFonts.latoTextTheme().copyWith(
              titleLarge: GoogleFonts.lato()
                  .copyWith(fontSize: 36, color: colorScheme.primary, fontWeight: FontWeight.bold)),
          colorScheme: colorScheme),
      home: const HomeScreen(),
    );
  }
}
